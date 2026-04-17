# Experimental Results

Detailed results from the paper: **"Utilizing Dynamic Context and Static Analysis for Agent-Based Automated Program Repair"** (IEEE Access, 2026).

---

## Benchmarks

| Benchmark | Language | Bugs | Description |
|-----------|----------|------|-------------|
| Defects4J v1.2 | Java | 395 | Classic real-world Java bugs |
| Defects4J v2.0 | Java | 265 | Extended bug dataset |
| **Total Defects4J** | **Java** | **660** | |
| SWE-Bench Lite | Python | 300 | Real GitHub issues |

---

## RQ1: Overall Effectiveness

### Defects4J Results (Perfect Fault Localization)

| Approach | v1.2 Correct | v2.0 Correct | Total Correct |
|----------|-------------|-------------|---------------|
| **Ours (GPT-4o)** | **187** | **178** | **365** |
| Ours (QwenCoder-32B)¹ | — | — | 250.3 ± 2.1 (best: 252) |
| Ours (CodeLlama-34B)¹ | — | — | 104.0 ± 3.6 (best: 108) |
| SRepair | 165 | 167 | 332 |
| ThinkRepair | — | — | 205 |
| D4C | — | — | 180 |
| ChatRepair | — | — | 162 |

¹ Open-source model results are reported as mean ± std across 3 independent runs.

**Improvement over baselines (GPT-4o):**
- vs. SRepair: **+9.9%** overall (+13.3% on v1.2, +6.6% on v2.0)
- vs. ThinkRepair: **+78.0%**
- vs. D4C: **+102.8%**
- vs. ChatRepair: **+125.3%**

**Unique bug fixes:** Our approach (GPT-4o) uniquely fixes **76 bugs** that no baseline system can repair — more than twice the unique contributions of SRepair (30), D4C (14), ThinkRepair (8), and ChatRepair (4). A core set of 59 bugs is fixed by all five systems.

### Patch Quality (Correctness Ratio)

| Model | Plausible | Correct | Correctness Ratio |
|-------|-----------|---------|-------------------|
| GPT-4o | 479 | 365 | 76.2% |
| QwenCoder-32B | 359.7 | 250.3 | 69.6% |
| CodeLlama-34B | 241 | 104.0 | 43.2% |

### SWE-Bench Lite Results (GPT-4o, Perfect FL)

**Overall: 87/300 bugs fixed (29.0% success rate)**

| Repository | Fixed/Total | Success Rate |
|------------|-------------|-------------|
| django | 36/114 | 31.6% |
| sympy | 20/77 | 26.0% |
| scikit-learn | 10/23 | 43.5% |
| pytest | 6/17 | 35.3% |
| matplotlib | 4/23 | 17.4% |
| sphinx | 4/16 | 25.0% |
| astropy | 3/6 | 50.0% |
| pylint | 2/6 | 33.3% |
| seaborn | 1/4 | 25.0% |
| xarray | 1/5 | 20.0% |
| flask | 0/3 | 0.0% |
| requests | 0/6 | 0.0% |

---

## RQ2: Iteration Analysis

### Iteration-wise Distribution of Correct Fixes

| Iteration | GPT-4o | QwenCoder-32B | CodeLlama-34B |
|-----------|--------|---------------|---------------|
| 1 | 343 (94.0%) | 240 (95.2%) | 92 (85.2%) |
| 2 | 16 | 12* | 8 |
| 3 | 4 | * | 5 |
| 4 | 2 | — | 1 |
| 5 | 0 | — | 2 |
| **Total** | **365** | **252** | **108** |

*QwenCoder-32B fixes 12 additional bugs across iterations 2–3.

### Key Findings

- **First-iteration success rate** is highest for stronger models: GPT-4o (94.0%), QwenCoder-32B (95.2%), CodeLlama-34B (85.2%).
- **Weaker models benefit more** from iterative context refinement: CodeLlama-34B generates 30.2% of its plausible patches at iteration ≥ 2, compared to only 9.2% for GPT-4o and 7.6% for QwenCoder-32B.
- The Context Updater agent **compensates for limited initial reasoning** in weaker models by progressively providing additional program analysis information.

---

## RQ3: Multi-Function Bug Performance

Multi-function bugs require coordinated modifications across multiple program locations.

### Correct Multi-Function Fixes by Number of Methods Changed

| Methods | GPT-4o | QwenCoder-32B | CodeLlama-34B | SRepair | D4C |
|---------|--------|---------------|---------------|---------|-----|
| 2 | 42 | 25 | 9 | 27 | 1 |
| 3 | 6 | 5 | — | 4 | 1 |
| 4 | 3 | 1 | 2 | 1 | — |
| 6 | 1 | 1 | — | — | — |
| 10 | 1 | 1 | — | — | — |
| **Total** | **53** | **33** | **11** | **32** | **2** |

**Multi-function fix rate** (relative to total correct fixes):
- GPT-4o: 14.5% (53/365)
- QwenCoder-32B: 13.1% (33/252)
- CodeLlama-34B: 10.2% (11/108)

Our approach is the **only system** that successfully repairs bugs requiring changes to 6 or more methods.

### SWE-Bench Lite Multi-Function Results (GPT-4o)

- 8 bugs requiring 2-function changes
- 2 bugs requiring 3-function changes

---

## RQ4: Ablation Study

Systematic ablation across all three model backends on Defects4J, isolating two architectural contributions: the **Overfitting Detector (OD)** and the **Context Updater (CU)**.

### Component Contribution (% of correct fixes attributed to each component)

| Component | GPT-4o | QwenCoder-32B | CodeLlama-34B |
|-----------|--------|---------------|---------------|
| Overfitting Detector (OD) | 4.1% | 12.0% | 4.5% |
| Context Updater (CU) | 6.0% | 6.7% | 15.4% |
| **Combined (OD + CU)** | **10.6%** | — | **23.0%** |

### Key Findings

- **OD is most valuable for mid-capability models:** QwenCoder relies on OD for 12.0% of its correct fixes, while GPT-4o and CodeLlama need it for ~4%.
- **CU is most valuable for weaker models:** CodeLlama relies on CU for 15.4% of its fixes vs. 6.0% for GPT-4o.
- **Complementary pattern:** OD compensates for mid-capability models' tendency to generate overfitting patches, while CU compensates for weaker models' limited initial reasoning.
- **Architectural gains increase as model capability decreases:** Combined improvement ranges from 10.6% (GPT-4o) to 23.0% (CodeLlama-34B).

---

## RQ5: Practical Fault Localization

Evaluation using **GZoltar + Ochiai** (spectrum-based FL) with top-10 suspicious methods, on 350 bugs where at least one ground-truth buggy method appears in the top-10 ranking.

| Condition | Plausible | Correct | Correctness Ratio |
|-----------|-----------|---------|-------------------|
| Perfect FL (350-bug subset) | 278 | 218 | 78.8% |
| GZoltar Top-10 | 232 | 152 | 65.5% |
| **Retention** | **83.5%** | **69.7%** | — |

The approach retains **69.7% of correct fixes** when switching from perfect to automated fault localization.

---

## Rerun Stability (Open-Source Models)

| Model | Run 1 | Run 2 | Run 3 | Mean | Std Dev | CV |
|-------|-------|-------|-------|------|---------|-----|
| QwenCoder-32B | 252 | 251 | 248 | 250.3 | 2.08 | 0.83% |
| CodeLlama-34B | 108 | 101 | 103 | 104.0 | 3.61 | 3.47% |

Low coefficients of variation confirm that performance differences are meaningful and not artifacts of stochastic variation.

---

## Computational Cost

| Metric | GPT-4o | QwenCoder-32B | CodeLlama-34B |
|--------|--------|---------------|---------------|
| Avg. time per fixed bug | 312.4s | 555.9s | 453.2s |
| Avg. input tokens (fixed) | 7,849 | 12,294 | 21,095 |
| Avg. output tokens (fixed) | 11,364 | 20,034 | 21,334 |
| Avg. time per unfixed bug | 835.2s | 1,149.2s | 1,239.0s |
| API cost per bug (GPT-4o) | $0.13 | N/A | N/A |
| Avg. iterations | 1.29 | — | — |

Open-source models (QwenCoder-32B, CodeLlama-34B) require **no API costs** — they run locally on 2× NVIDIA RTX A6000 GPUs via vLLM.

---

## Component-Level Mapping Analysis

Among the 80 multi-function bugs in Defects4J with ≥2 buggy methods and ≥2 failing tests:
- **38 bugs (47.5%)** were decomposed into multiple independent repair components
- **42 bugs (52.5%)** were identified as single cohesive units

Decomposition breakdown of the 38 decomposed bugs:
- 31 bugs → 2 independent components
- 4 bugs → 3 components
- 2 bugs → 4 components
- 1 bug → 5 components

---

## Results Directory Structure

Experiment results (patch JSONs) are stored in the repository:

```
results_gpt4o/
└── results/          # 659 patch JSON files (GPT-4o)

results_open_source/
├── codellama/
│   ├── results_codellama_exp_1/  # Run 1 (660 files)
│   ├── results_codellama_exp_2/  # Run 2 (660 files)
│   └── results_codellama_exp_3/  # Run 3 (660 files)
└── qwen/
    ├── results_qwen_exp_1/   # Run 1 (660 files)
    ├── results_qwe_exp_2/    # Run 2 (660 files, naming typo)
    └── results_qwen_exp_3/   # Run 3 (660 files)
```

Each result file is a JSON containing the generated patch, plausibility status, and correctness assessment for a single bug.

> **Note:** Execution traces (~1.5 GB total) are excluded from the repository due to size constraints. They are generated automatically when running the tool (saved to the `--traces-dir` path).
