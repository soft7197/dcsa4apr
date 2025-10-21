#!/usr/bin/env python3
"""
Retroactive overfitting detection + refinement for passing CodeLlama patches.

Reads:  results_codellama_exp_3/  +  traces_codellama_exp_3/
Writes: results_codellama_exp_3_overfitting/

Output format is IDENTICAL to QWE experiment output:
  - Not overfitting / refinement failed  -> *_patch.json unchanged
  - Overfitting + refinement succeeded   -> *_patch.json patch replaced with
      refined patch that has is_refinement=True, refinement_attempt=N,
      original_hypothesis="..."  (exactly what controller._refine_overfitting_patch adds)

No extra files, no extra fields anywhere else.
"""

import json
import os
import re
import glob
import shutil
import sys
import time
import subprocess
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from utils.json_parser import extract_json
from execution.executor import PatchExecutor

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not found.")
    sys.exit(1)

logging.basicConfig(level=logging.WARNING)   # suppress executor noise

# ── Config — overridable via CLI ───────────────────────────────────────────────
RESULTS_DIR             = "results_codellama_exp_3"
TRACES_DIR              = "traces_codellama_exp_3"
OUTPUT_DIR              = "results_codellama_exp_3_overfitting"
PROXY_URL               = "http://localhost:8000/v1"
MODEL                   = "codellama/CodeLlama-34b-Instruct-hf"
MAX_REFINEMENT_ATTEMPTS = 1
MAX_RETRIES             = 1
RETRY_DELAY             = 1


# ── Prompts — exact copies from controller ────────────────────────────────────

def _create_overfitting_detection_prompt(patch: dict, failing_tests: dict) -> str:
    prompt = f"""## Semantic Validation - Overfitting Detection

### TASK:
Analyze if this patch is overfitting to the test cases or if it genuinely fixes the root cause.

### ORIGINAL BUGGY CODE:
```java
{patch.get('original_code', '')}
```

### PATCHED CODE:
```java
{patch.get('fixed_method', '')}
```

### PATCH HYPOTHESIS:
{patch.get('hypothesis', 'No hypothesis provided')}

### FAILING TESTS THAT NOW PASS:
"""
    for test in list(failing_tests.keys())[:5]:
        test_info = failing_tests.get(test, {})
        prompt += f"""
- Test:
```java
{failing_tests[test].get('src', test)}
```
  Original Error: {test_info.get('error_type', 'Unknown')}
"""
    prompt += """

### ANALYSIS REQUIRED:
1. Does the patch fix the root cause or just make tests pass?
2. Are there hardcoded values that match test expectations?
3. Does the logic handle general cases or only test inputs?
4. Are there missing edge cases that tests don't cover?
5. Could this break other functionality not covered by tests?


Find the answers for the questions above in yes:why and no:why format.

### Return JSON:
{
    "is_overfitting": true/false,
    "can_be_improved": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of specific overfitting issues found"],
    "root_cause_fixed": true/false,
    "refinement_suggestions": ["specific suggestions to fix overfitting or improve"],
    "semantic_correctness": "answers to ANALYSIS REQUIRED questions"
}

### IMPORTANT: 1. while answering the 4th question, think beyond the current provided test. Because current test cases are not enough to test the buggy methods.
               2. Improvements should be specific to the current patch and not generic suggestions.If no specific suggestions, return can_be_improved as false.
"""
    return prompt


def _create_refinement_prompt(patch: dict, overfitting_analysis: dict, attempt: int) -> str:
    prompt = f"""## Patch Refinement - Attempt {attempt}

### CONTEXT:
The following patch passes all tests but was detected as overfitting or improvement suggestions were found for this patch.
It needs to be refined to fix the root cause while maintaining test passing.

### ORIGINAL BUGGY CODE:
```java
{patch.get('original_code', '')}
```

### CURRENT PATCH (OVERFITTING):
```java
{patch.get('fixed_method', '')}
```

### OVERFITTING ISSUES DETECTED:
"""
    for issue in overfitting_analysis.get('issues', []):
        prompt += f"- {issue}\n"

    prompt += """

### REFINEMENT or IMPROVEMENT SUGGESTIONS:
"""
    for suggestion in overfitting_analysis.get('refinement_suggestions', []):
        prompt += f"- {suggestion}\n"

    prompt += """

### TASK:
Create a refined patch that:
1. Fixes the root cause identified in the analysis
2. Maintains test-passing behavior
3. Handles general cases, not just test inputs
4. Includes proper edge case handling
5. Avoids the overfitting patterns identified

### REQUIREMENTS:
- The refined patch MUST still pass all the original failing tests
- The logic should be general and correct, not test-specific
- Include proper null checks and boundary conditions
- Fix the actual root cause, not symptoms

Provide JSON response:
{
    "hypothesis": "Explanation of the refined fix addressing overfitting",
    "fixed_method": "Complete refined method code"
}
"""
    return prompt


_GENERATOR_SYSTEM_PROMPT = """You are an expert software engineer specializing in automated program repair.
Your task is to fix bugs in code based on test failures and error messages and the provided context.

Guidelines:
1. Analyze the error carefully to understand the root cause
2. Learn from previous failed attempts - don't repeat the same mistakes
3. Ensure the fixed code is syntactically correct
4. Focus on the failing test requirements
5. Use the provided context effectively

CRITICAL: You MUST respond with valid JSON only. Do not include any text, markdown, or explanation outside the JSON object.
Your response must be parseable by json.loads(). No trailing commas, no comments, no markdown code fences.

For single-method fixes, use this format:
{
    "hypothesis": "Brief explanation of the bug cause and fix approach",
    "changes": "the main changes in +/- format (short and understandable)",
    "fixed_method": "Complete fixed method code without comments"
}
"""


# ── JSON parsing — same as controller._detect_overfitting ────────────────────

def _parse_overfitting_response(raw: str) -> dict:
    analysis = extract_json(raw, fallback=None)
    if analysis is None:
        lower = raw.lower()
        kw = {}
        for key in ('is_overfitting', 'can_be_improved'):
            m = re.search(rf'["\']?{key}["\']?\s*[:=]\s*(true|false)', lower)
            if m:
                kw[key] = (m.group(1) == 'true')
        if kw:
            analysis = {
                'is_overfitting':  kw.get('is_overfitting', False),
                'can_be_improved': kw.get('can_be_improved', False),
                'confidence': 0.5,
                'issues': ['parsed from malformed response via keyword scan'],
            }
    if not isinstance(analysis, dict):
        analysis = {'is_overfitting': False, 'can_be_improved': False,
                    'confidence': 0, 'issues': []}
    return analysis


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _detect_overfitting(client: OpenAI, patch: dict, failing_tests: dict) -> dict:
    """Same structure as controller._detect_overfitting."""
    prompt = _create_overfitting_detection_prompt(patch, failing_tests)
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are an expert at detecting overfitting in program repair patches."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=600,
                n=1,
            )
            return _parse_overfitting_response(response.choices[0].message.content)
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return {'is_overfitting': False, 'can_be_improved': False,
                        'confidence': 0, 'issues': [], 'error': str(e)}


def _generate_refined_patches(client: OpenAI, prompt: str) -> list:
    """Same as generator_agents.generate_patch (n=10 like controller uses for refinement)."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            response_format={"type": "json_object"},
            n=10,
        )
        patches = []
        seen = set()
        for choice in response.choices:
            p = extract_json(choice.message.content, fallback=None)
            if not isinstance(p, dict):
                continue
            fixed = p.get("fixed_method", "").strip()
            fixed_norm = re.sub(r'\s+', ' ', fixed)
            if not fixed_norm or fixed_norm in seen:
                continue
            seen.add(fixed_norm)
            p['diff'] = p.get('changes', ' ')
            patches.append(p)
        return patches
    except Exception as e:
        print(f"    [generate error] {e}")
        return []


# ── Defects4J checkout — same as main.checkout_defects4j_bug ─────────────────

def checkout_defects4j_bug(bug_id: str) -> str | None:
    project, bug_num = bug_id.split('-')
    working_dir = f"/tmp/d4j_{bug_id}_{int(time.time())}"
    try:
        r = subprocess.run(
            ["defects4j", "checkout", "-p", project, "-v", f"{bug_num}b", "-w", working_dir],
            capture_output=True, text=True, timeout=120
        )
        if r.returncode != 0:
            return None
        return working_dir if os.path.exists(working_dir) else None
    except Exception:
        return None


# ── Trace helpers ─────────────────────────────────────────────────────────────

def get_failing_tests(bug_id: str) -> dict:
    """Returns {test_name: {"src": test_name, "error_type": "TestFailure"}}."""
    iter_dir = os.path.join(TRACES_DIR, bug_id, "iterations")
    if not os.path.isdir(iter_dir):
        return {}
    tests = []
    for comp in sorted(os.listdir(iter_dir)):
        info_file = os.path.join(iter_dir, comp, "component_info.json")
        if os.path.isfile(info_file):
            try:
                tests.extend(json.load(open(info_file)).get("failing_tests", []))
            except Exception:
                pass
    seen = set()
    unique = [t for t in tests if not (t in seen or seen.add(t))]
    return {t: {"src": t, "error_type": "TestFailure"} for t in unique}


def _result_patches_flat(result_data: dict) -> list:
    """Return a flat list of patch dicts from result_data['patches']."""
    out = []
    for group in result_data.get("patches", []):
        if isinstance(group, list):
            out.extend(p for p in group if isinstance(p, dict))
        elif isinstance(group, dict):
            out.append(group)
    return out


def find_passing_patch(bug_id: str, result_data: dict) -> dict | None:
    """Find first patch with execution_result.status==success in traces.

    Supplements missing line_numbers from result_data when the trace patch
    doesn't carry them (common in CodeLlama traces).
    """
    result_flat = _result_patches_flat(result_data)

    iter_dir = os.path.join(TRACES_DIR, bug_id, "iterations")
    if os.path.isdir(iter_dir):
        for comp in sorted(os.listdir(iter_dir)):
            comp_dir = os.path.join(iter_dir, comp)
            if not os.path.isdir(comp_dir):
                continue
            for iter_file in sorted(glob.glob(os.path.join(comp_dir, "iteration_*.json"))):
                try:
                    for p in json.load(open(iter_file)).get("patches", []):
                        if (isinstance(p, dict) and
                                p.get("execution_result", {}).get("status") == "success"):
                            # Supplement missing line_numbers from result_data
                            if not p.get("line_numbers"):
                                for rp in result_flat:
                                    if (rp.get("method_name") == p.get("method_name") and
                                            rp.get("file_path") == p.get("file_path") and
                                            rp.get("line_numbers")):
                                        p = dict(p)  # copy so we don't mutate trace
                                        p["line_numbers"] = rp["line_numbers"]
                                        break
                            return p
                except Exception:
                    pass

    # Fallback: use first patch from result_data (already has all fields)
    if result_flat:
        return result_flat[0]
    return None


def replace_patch_in_result(result_data: dict, original_patch: dict,
                             refined_patch: dict) -> None:
    """
    Replace the original passing patch with the refined patch inside
    result_data['patches'], same position. This mirrors what the controller
    does at results[idx-1]['patch'] = refined_patch.
    """
    patches = result_data.get("patches", [])
    for group in patches:
        group_list = group if isinstance(group, list) else [group]
        for idx, p in enumerate(group_list):
            if not isinstance(p, dict):
                continue
            if (p.get("method_name") == original_patch.get("method_name") and
                    p.get("file_path")   == original_patch.get("file_path") and
                    p.get("iteration")   == original_patch.get("iteration")):
                group_list[idx] = refined_patch
                return
    # Fallback: prepend to first group if no match found
    if patches:
        first = patches[0]
        if isinstance(first, list):
            first.insert(0, refined_patch)


# ── Per-bug worker ─────────────────────────────────────────────────────────────

_print_lock = threading.Lock()

def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def process_bug(client: OpenAI, idx: int, total: int,
                result_file: str, result_data: dict) -> None:
    bug_id = result_data["bug_id"]
    prefix = f"[{idx:3d}/{total}] {bug_id}"

    failing_tests_dict = get_failing_tests(bug_id)
    failing_tests_list = list(failing_tests_dict.keys())

    if not failing_tests_dict:
        _log(f"{prefix}  SKIP (no tests in trace)")
        shutil.copy2(result_file, os.path.join(OUTPUT_DIR, os.path.basename(result_file)))
        return

    passing_patch = find_passing_patch(bug_id, result_data)
    if passing_patch is None:
        _log(f"{prefix}  SKIP (no passing patch)")
        shutil.copy2(result_file, os.path.join(OUTPUT_DIR, os.path.basename(result_file)))
        return

    # ── Overfitting detection ──────────────────────────────────────────────
    analysis = _detect_overfitting(client, passing_patch, failing_tests_dict)
    is_overfitting  = bool(analysis.get("is_overfitting",  False))
    can_be_improved = bool(analysis.get("can_be_improved", False))

    if not (is_overfitting or can_be_improved):
        _log(f"{prefix}  OK")
        shutil.copy2(result_file, os.path.join(OUTPUT_DIR, os.path.basename(result_file)))
        return

    label = "OVERFIT" if is_overfitting else "IMPROVE"
    conf  = analysis.get("confidence", 0)
    _log(f"{prefix}  {label}  conf={conf:.2f}  → refining...")

    # ── Checkout bug for test execution ───────────────────────────────────
    project_path = checkout_defects4j_bug(bug_id)
    if not project_path:
        _log(f"{prefix}  WARN: checkout failed, keeping original patch")
        shutil.copy2(result_file, os.path.join(OUTPUT_DIR, os.path.basename(result_file)))
        return

    executor = PatchExecutor(project_path)
    refined_successfully = False

    try:
        for attempt in range(1, MAX_REFINEMENT_ATTEMPTS + 1):
            ref_prompt = _create_refinement_prompt(passing_patch, analysis, attempt)
            candidates = _generate_refined_patches(client, ref_prompt)

            for candidate in candidates:
                candidate["file_path"]          = passing_patch.get("file_path")
                candidate["method_name"]         = passing_patch.get("method_name")
                candidate["class_name"]          = passing_patch.get("class_name")
                candidate["iteration"]           = passing_patch.get("iteration", 1)
                candidate["is_refinement"]       = True
                candidate["refinement_attempt"]  = attempt
                candidate["original_hypothesis"] = passing_patch.get("hypothesis")
                candidate["line_numbers"]        = passing_patch.get("line_numbers") or []

                exec_result = executor.execute_patch(candidate, failing_tests_list)
                if exec_result["status"] == "success":
                    replace_patch_in_result(result_data, passing_patch, candidate)
                    refined_successfully = True
                    break

            if refined_successfully:
                _log(f"{prefix}  attempt {attempt}  REFINED ✓")
                break
            else:
                _log(f"{prefix}  attempt {attempt}  failed")

    except Exception as e:
        _log(f"{prefix}  ERROR: {e}")
    finally:
        shutil.rmtree(project_path, ignore_errors=True)

    if not refined_successfully:
        _log(f"{prefix}  all refinement attempts failed — keeping original patch")

    out_path = os.path.join(OUTPUT_DIR, os.path.basename(result_file))
    with open(out_path, "w") as fp:
        json.dump(result_data, fp, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global RESULTS_DIR, TRACES_DIR, OUTPUT_DIR, MODEL

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bugs", nargs="+", default=[],
                        help="Bug IDs to process (e.g. Chart-11 Cli-12). Default: all.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4).")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="Source results directory.")
    parser.add_argument("--traces-dir", default=TRACES_DIR,
                        help="Source traces directory.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory.")
    parser.add_argument("--model", default=MODEL,
                        help="LLM model name.")
    args = parser.parse_args()
    bug_filter = set(args.bugs)

    RESULTS_DIR = args.results_dir
    TRACES_DIR  = args.traces_dir
    OUTPUT_DIR  = args.output_dir
    MODEL       = args.model

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = OpenAI(base_url=PROXY_URL, api_key="unused")
    print(f"Checking proxy at {PROXY_URL} ...")
    try:
        client.models.list()
        print("Proxy OK\n")
    except Exception as e:
        print(f"ERROR: Cannot reach proxy: {e}")
        sys.exit(1)

    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_patch.json")))
    success_files = []
    other_files   = []
    for f in result_files:
        try:
            d = json.load(open(f))
            (success_files if d.get("success") else other_files).append((f, d))
        except Exception:
            other_files.append((f, None))

    if bug_filter:
        success_files = [(f, d) for f, d in success_files if d["bug_id"] in bug_filter]

    # Copy all non-success files unchanged (fast, no parallelism needed)
    for f, _ in other_files:
        shutil.copy2(f, os.path.join(OUTPUT_DIR, os.path.basename(f)))
    orig_summary = os.path.join(RESULTS_DIR, "repair_summary.json")
    if os.path.isfile(orig_summary):
        shutil.copy2(orig_summary, os.path.join(OUTPUT_DIR, "repair_summary.json"))

    total = len(success_files)
    print(f"Success bugs to process : {total}  (workers={args.workers})\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_bug, client, i, total, f, d): (f, d)
            for i, (f, d) in enumerate(success_files, 1)
        }
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                f, d = futures[fut]
                _log(f"UNHANDLED ERROR for {d.get('bug_id','?')}: {exc}")

    print("\n=== DONE ===")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
