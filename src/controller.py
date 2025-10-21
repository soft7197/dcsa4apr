# Enhanced controller.py with overfitting detection and refinement
"""
Enhanced controller that detects overfitting and refines patches.
Adds semantic validation after test passing to prevent overfitting.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import json
from dataclasses import dataclass
import logging
import os
from datetime import datetime
import re

from src.tools.fault_localization import FaultLocation
from src.utils.json_parser import extract_json

ready_components_path = 'data/connected_components'

@dataclass
class BugFixingConfig:
    max_iterations: int = 5
    max_hypothesis_pool_size: int = 10
    knowledge_base_token_limit: int = 8000
    use_docker: bool = True
    fl_tool: str = "gzoltar"  # or "perfect"
    llm_model: str = "gpt-4o"
    temperature: float = 0.2
    perfect_fl_file: str = "data/perfect_fl.json"
    parallel_components: bool = False  # For multi-component bugs
    enable_smart_resolution: bool = True
    enable_caching: bool = True
    # New configuration for overfitting detection
    enable_overfitting_detection: bool = True
    max_refinement_attempts: int = 1
    semantic_validation_model: str = "gpt-4o"


class MainController:
    def __init__(self, config: BugFixingConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        # Initialize monitor storage
        self.monitors = {}
        
        # Load perfect FL data if needed
        self.perfect_fl_data = None
        if self.config.fl_tool == "perfect":
            self._load_perfect_fl_data()
        
        # Initialize components (will be populated by initialize_components)
        self.fault_localizer = None
        self.prompt_maker = None
        self.generator = None
        self.executor = None
        self.hypothesis_pool = None
        self.hypothesis_updater = None
        self.context_manager = None
        self.tools = {}
        
        # Metrics
        self.metrics = {
            'total_iterations': 0,
            'successful_patches': 0,
            'failed_patches': 0,
            'time_spent': 0,
            'overfitting_detected': 0,
            'successful_refinements': 0,
            'failed_refinements': 0
        }
        
        # Current bug info for context
        self.current_bug_info = None
        
        # Cache for tool results
        self.tool_cache = {} if config.enable_caching else None
        
        # Track refinement history
        self.refinement_history = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('BugFixer')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('bugfix.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _detect_overfitting(self, patch: Dict, bug_info: Dict, failing_tests: List[str]) -> Tuple[bool, Dict]:
        """
        Use LLM to detect if a patch is overfitting to tests.
        
        Args:
            patch: The patch that passed tests
            bug_info: Bug information
            failing_tests: List of failing tests
            
        Returns:
            Tuple of (is_overfitting, analysis_details)
        """
        self.logger.info("Performing semantic validation to detect overfitting...")
        
        # Get monitor
        monitor = self.monitors.get(bug_info.get('bug_id'))
        
        # Build prompt for overfitting detection
        prompt = self._create_overfitting_detection_prompt(patch, bug_info, failing_tests)
        
        if monitor:
            monitor.log_prompt(
                prompt=prompt,
                prompt_type="overfitting_detection",
                metadata={
                    'patch_hypothesis': patch.get('hypothesis', ''),
                    'method_name': patch.get('method_name', ''),
                    'iteration': patch.get('iteration', 0)
                }
            )
        
        # Call LLM for semantic analysis
        try:


            # from openai import OpenAI
            # client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
            # response = client.chat.completions.create(
            #     model="codellama/CodeLlama-34b-Instruct-hf",
            #     messages=[
            #         {"role": "system", "content": "You are an expert at detecting overfitting in program repair patches. You MUST respond with valid JSON only. Do not include any text, markdown, or explanation outside the JSON object. Your response must be parseable by json.loads(). Use this format: {\"is_overfitting\": true/false, \"can_be_improved\": true/false, \"confidence\": 0.0-1.0, \"issues\": [\"list of issues\"], \"root_cause_fixed\": true/false, \"refinement_suggestions\": [\"suggestions\"], \"semantic_correctness\": \"analysis answers\"}"},
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=0.1,
            #     response_format={"type": "json_object"}
            # )



            response = self.context_manager.client.chat.completions.create(
                model=self.config.semantic_validation_model,
                messages=[
                    {"role": "system", "content": "You are an expert at detecting overfitting in program repair patches."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content

            # Strategy 1: standard multi-method JSON extraction
            analysis = extract_json(raw_content, fallback=None)

            # Strategy 2: keyword regex scan when JSON extraction fails completely
            if analysis is None:
                lower = raw_content.lower()
                kw_result = {}
                for key in ('is_overfitting', 'can_be_improved'):
                    m = re.search(rf'["\']?{key}["\']?\s*[:=]\s*(true|false)', lower)
                    if m:
                        kw_result[key] = (m.group(1) == 'true')
                if kw_result:
                    self.logger.warning("Overfitting response was not valid JSON; recovered via keyword scan")
                    analysis = {
                        'is_overfitting': kw_result.get('is_overfitting', False),
                        'can_be_improved': kw_result.get('can_be_improved', False),
                        'confidence': 0.5,
                        'issues': ['parsed from malformed response via keyword scan'],
                    }

            # Strategy 3: safe defaults when all extraction fails
            if not isinstance(analysis, dict):
                self.logger.warning("Could not extract overfitting analysis; defaulting to not-overfitting")
                analysis = {'is_overfitting': False, 'can_be_improved': False, 'confidence': 0, 'issues': []}
            
            if monitor:
                monitor.log_llm_response(
                    response=json.dumps(analysis),
                    response_type="overfitting_analysis",
                    metadata={
                        'is_overfitting': analysis.get('is_overfitting', False),
                        'confidence': analysis.get('confidence', 0),
                        'issues_count': len(analysis.get('issues', []))
                    }
                )
            
            can_be_improved, is_overfitting = analysis.get('can_be_improved', False), analysis.get('is_overfitting', False)
            
            if is_overfitting or can_be_improved:
                self.logger.warning(f"🔍 Overfitting or improvement suggestions detected! Confidence: {analysis.get('confidence', 0)}")
                self.logger.info(f"Issues: {', '.join(analysis.get('issues', []))}")
                self.metrics['overfitting_detected'] += 1
            else:
                self.logger.info("✓ Patch appears semantically correct (no overfitting detected)")
            
            return can_be_improved, is_overfitting, analysis
            
        except Exception as e:
            self.logger.error(f"Error in overfitting detection: {e}")
            # On error, assume not overfitting to avoid blocking progress
            return False, False, {"error": str(e), "skipped": True}
    
    def _create_overfitting_detection_prompt(self, patch: Dict, bug_info: Dict, failing_tests: List[str]) -> str:
        """Create prompt for overfitting detection."""
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
        
        # Add test information
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
    
    def _refine_overfitting_patch(self, patch: Dict, overfitting_analysis: Dict, 
                                 bug_info: Dict, failing_tests: List[str], 
                                 refinement_attempt: int) -> Optional[Dict]:
        
        
        last_correct_patch= None
        """
        Refine a patch that was detected as overfitting.
        
        Args:
            patch: The overfitting patch
            overfitting_analysis: Analysis from overfitting detection
            bug_info: Bug information
            failing_tests: Failing tests
            refinement_attempt: Current refinement attempt number
            
        Returns:
            Refined patch or None if refinement fails
        """
        self.logger.info(f"Refining overfitting patch - Attempt {refinement_attempt}/{self.config.max_refinement_attempts}")
        
        # Get monitor
        monitor = self.monitors.get(bug_info.get('bug_id'))
        
        # Build refinement prompt
        prompt = self._create_refinement_prompt(patch, overfitting_analysis, refinement_attempt)
        
        if monitor:
            monitor.log_prompt(
                prompt=prompt,
                prompt_type="refinement",
                metadata={
                    'refinement_attempt': refinement_attempt,
                    'original_hypothesis': patch.get('hypothesis', ''),
                    'overfitting_issues': overfitting_analysis.get('issues', [])
                }
            )
        
        # Generate refined patch
        try:
            gen_result = self.generator.generate_patch(prompt, patch.get('fixed_method', ''), 1)
            refined_patches = gen_result['patches']

            for refined_patch in refined_patches:
                # Add metadata
                refined_patch['file_path'] = patch.get('file_path')
                refined_patch['method_name'] = patch.get('method_name')
                refined_patch['class_name'] = patch.get('class_name')
                refined_patch['iteration'] = patch.get('iteration', 0)
                refined_patch['is_refinement'] = True
                refined_patch['refinement_attempt'] = refinement_attempt
                refined_patch['original_hypothesis'] = patch.get('hypothesis')
                refined_patch['line_numbers'] = patch.get('line_numbers', [])
                
                if monitor:
                    monitor.log_hypothesis(refined_patch)
                
                # Test refined patch
                execution_result = self.executor.execute_patch(refined_patch, failing_tests)
                
                if monitor:
                    monitor.log_execution_result(execution_result)
                
                if execution_result["status"] == "success":
                    self.logger.info(f"✓ Refined patch passes tests!")
                    last_correct_patch = refined_patch
                    # Check if still overfitting
                    still_can_be_improved, is_still_overfitting, new_analysis = self._detect_overfitting(
                        refined_patch, bug_info, failing_tests
                    )
                    
                    if not is_still_overfitting:
                        self.logger.info(f"✅ SUCCESSFUL REFINEMENT! Patch no longer overfitting!")
                        self.metrics['successful_refinements'] += 1
                        return refined_patch, True
                    else:
                        self.logger.warning(f"Refined patch still shows overfitting")
                        # Continue to next refinement candidate
                        continue
                else:
                    self.logger.warning(f"Refined patch failed tests: {execution_result.get('error_type')}")
            
            self.metrics['failed_refinements'] += 1
            return last_correct_patch, False
            
        except Exception as e:
            self.logger.error(f"Error during refinement: {e}")
            self.metrics['failed_refinements'] += 1
            return None
    
    def _create_refinement_prompt(self, patch: Dict, overfitting_analysis: Dict, attempt: int) -> str:
        """Create prompt for patch refinement."""
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
        
        prompt += f"""

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
    
    def _process_single_method(self, buggy_method: FaultLocation,
                               failing_tests: List[str], bug_info: Dict,
                               sibling_fixes: List[Dict] = None) -> Optional[Dict]:
        """Enhanced process for single method with overfitting detection and refinement."""
        self.logger.info(f"Processing single method: {buggy_method.method_name}")

        # Get monitor
        monitor = self.monitors.get(bug_info.get('bug_id'))

        # Build knowledge base for single method
        knowledge_base = self._build_knowledge_base(
            [buggy_method],
            failing_tests,
            bug_info
        )

        # Initialize empty dynamic context
        dynamic_context = {}

        # Main repair loop
        for iteration in range(5):
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
            self.metrics['total_iterations'] += 1

            if monitor:
                monitor.log_iteration_start(iteration + 1)

            # Update dynamic context after first failure
            if iteration > 0:
                self.logger.info("Requesting LLM decision for context update...")

                # Get LLM decision on what to retrieve
                decision = self.context_manager.decide_context_update(
                    buggy_method=buggy_method.buggy_code if hasattr(buggy_method, 'buggy_code') else '',
                    kb=knowledge_base,
                    tried_hypotheses=self.hypothesis_pool.get_recent_hypotheses(),
                    current_context=dynamic_context,
                    retrieval_history=self.context_manager.state.retrieval_history
                )

                if monitor:
                    monitor.log_context_decision(decision)

                self.logger.info(f"LLM decided to use tools: {[cmd['tool'] for cmd in decision.get('tool_commands', [])]}")

                # Execute LLM decisions with validated commands
                dynamic_context = self.context_manager.execute_context_decisions(
                    current_context=dynamic_context,
                    decision=decision,
                    tools=self.tools,
                    bug_info={
                        'bug_id': bug_info.get('bug_id'),
                        'buggy_method': buggy_method.buggy_code,
                        'buggy_method_name': buggy_method.method_name,
                        'buggy_class': buggy_method.class_name,
                        'buggy_file_path': buggy_method.file_path,
                        'failing_tests': failing_tests,
                        'project_path': self.executor.project_path
                    }
                )

                # Extract and log tool execution logs
                if monitor and '_tool_logs' in dynamic_context:
                    monitor.log_tool_executions(dynamic_context['_tool_logs'])

                # Remove metadata key before using context
                dynamic_context.pop('_tool_logs', None)

                self.logger.info(f"Dynamic context updated: {len(dynamic_context)} items")

            # Generate prompt for single method
            # From iteration 2 onward, include sibling component fixes as context
            sibling_fixes = sibling_fixes if iteration >= 1 else []
            if sibling_fixes:
                sibling_fixes = sibling_fixes[:1]
            prompt = self._create_single_method_prompt(
                buggy_method=buggy_method,
                knowledge_base=knowledge_base,
                dynamic_context=dynamic_context,
                iteration=iteration + 1,
                recent_hypotheses=self.hypothesis_pool.get_recent_hypotheses() if iteration > 0 else [],
                sibling_fixes=sibling_fixes if iteration >= 1 else [],
                prohibited_symbols=self.hypothesis_pool.get_prohibited_symbols()
            )
            #print(prompt)

            # Generate patch (now returns {'patches': [...], 'usage': {...}})
            gen_result = self.generator.generate_patch(prompt, buggy_method.buggy_code if hasattr(buggy_method, 'buggy_code') else '', 2)
            patch_hypothesises = gen_result['patches']
            gen_usage = gen_result.get('usage', {})

            if monitor:
                monitor.log_generation(prompt, gen_usage)

            # Add metadata
            for patch_hypothesis in patch_hypothesises:
                patch_hypothesis['file_path'] = buggy_method.file_path
                patch_hypothesis['method_name'] = buggy_method.method_name
                patch_hypothesis['class_name'] = buggy_method.class_name
                patch_hypothesis['iteration'] = iteration + 1
                patch_hypothesis['is_multi_method'] = False
                patch_hypothesis['line_numbers'] = knowledge_base.get('buggy_methods', [{}])[0].get('lines', [])
                patch_hypothesis['original_code'] = buggy_method.buggy_code if hasattr(buggy_method, 'buggy_code') else ''

                if monitor:
                    monitor.log_hypothesis(patch_hypothesis)

            # Execute patches
            # Track best patch by fewest failing tests (Bug 4 fix: use execution_result, not patch dict)
            best_exec_result = None
            the_best_patch_for_insight = patch_hypothesises[0] if patch_hypothesises else None
            results = [{f'{iteration}-{idx}': None, 'patch': None} for idx in range(1, len(patch_hypothesises) + 1)]
            failed_patches_this_iter = []  # collect all failures for pool (Bug 2 fix)
            for idx, patch_hypothesis in enumerate(patch_hypothesises, 1):
                execution_result = self.executor.execute_patch(
                    patch_hypothesis,
                    failing_tests
                )

                if monitor:
                    monitor.log_patch_result(patch_hypothesis, execution_result)
                results[idx - 1][f'{iteration}-{idx}'] = execution_result
                results[idx - 1]['patch'] = patch_hypothesis
                # Check result
                if execution_result["status"] == "success":
                    self.logger.info(f"✓ Patch {idx} passes tests at iteration {iteration + 1}!")
                    # NEW: Check for overfitting if enabled
                    if self.config.enable_overfitting_detection:
                        can_be_improved, is_overfitting, overfitting_analysis = self._detect_overfitting(
                            patch_hypothesis, bug_info, failing_tests
                        )

                        if is_overfitting or can_be_improved:
                            # Try to refine; the original passing patch is always the safety net
                            refined_successfully = False
                            for refinement_attempt in range(1, self.config.max_refinement_attempts + 1):
                                refined_patch, fully_refined = self._refine_overfitting_patch(
                                    patch_hypothesis,
                                    overfitting_analysis,
                                    bug_info,
                                    failing_tests,
                                    refinement_attempt
                                )

                                if refined_patch and fully_refined:
                                    self.logger.info(f"✅ SUCCESSFULLY REFINED PATCH!")
                                    self.metrics["successful_patches"] += 1
                                    results[idx - 1]['patch'] = refined_patch
                                    refined_successfully = True
                                    break

                            if not refined_successfully:
                                # Refinement failed or gave only partial result — keep the
                                # original passing patch; never discard a passing patch.
                                self.logger.warning(
                                    f"Could not fully refine patch after "
                                    f"{self.config.max_refinement_attempts} attempts; "
                                    f"keeping original passing patch"
                                )
                                results[idx - 1]['patch'] = patch_hypothesis
                                self.metrics["successful_patches"] += 1
                        else:
                            # Patch is semantically correct
                            self.logger.info(f"✅ SUCCESS with semantically correct patch {idx}!")
                            self.metrics["successful_patches"] += 1
                    else:
                        # Overfitting detection disabled, accept patch
                        self.logger.info(f"✅ SUCCESS with patch {idx} (semantic validation disabled)")
                        self.metrics["successful_patches"] += 1
                        if monitor:
                            monitor.finish_iteration()
                        return patch_hypothesis
                else:
                    self.logger.warning(f"✗ Failed (patch {idx}): {execution_result.get('error_type')}")
                    self.metrics["failed_patches"] += 1
                    # Bug 4 fix: compare using execution_result's failing_tests, not the patch dict
                    n_failing = len(execution_result.get('failing_tests', []))
                    n_best = len(best_exec_result.get('failing_tests', [])) if best_exec_result else float('inf')
                    if n_failing < n_best:
                        best_exec_result = execution_result
                        the_best_patch_for_insight = patch_hypothesis
                    # Bug 2 fix: collect every failed patch for pool feedback
                    patch_hypothesis['execution_result'] = execution_result
                    failed_patches_this_iter.append(patch_hypothesis)

            # Finish iteration tracking
            if monitor:
                monitor.finish_iteration()

            # Check if any patch succeeded
            plausible_patches = []
            for entry in results:
                if entry[next(iter(entry))]['status']=='success':
                    plausible_patches.append(entry['patch'])
            if len(plausible_patches)>0:
                return plausible_patches

            else:
                # Bug 2+3 fix: add ALL unique failed patches to pool so the prompt
                # accumulates diverse failure signals across iterations.
                final_exec = best_exec_result if best_exec_result else execution_result
                for failed_patch in failed_patches_this_iter:
                    self.hypothesis_updater.analyze_and_update(
                        failed_patch,
                        failed_patch['execution_result'],
                        self.hypothesis_pool
                    )
                insights = self.hypothesis_updater.analyze_and_update(
                    the_best_patch_for_insight,
                    final_exec,
                    self.hypothesis_pool
                )
            if self._should_stop_early(iteration, insights):
                self.logger.info("Stopping early based on insights")
                break

        self.logger.warning(
            f"Failed to fix single method after {self.config.max_iterations} iterations "
            f"and {len(patch_hypothesises)} candidates"
        )
        return None
    
    def _process_multi_method(self, buggy_methods: List[FaultLocation],
                         failing_tests: List[str], bug_info: Dict,
                         sibling_fixes: List[Dict] = None) -> Optional[Dict]:
        """Enhanced multi-method processing with overfitting detection."""
        self.logger.info(f"Processing {len(buggy_methods)} interdependent buggy methods")

        # Get monitor
        monitor = self.monitors.get(bug_info.get('bug_id'))

        # Build knowledge base for ALL methods
        knowledge_base = self._build_knowledge_base(
            buggy_methods,
            failing_tests,
            bug_info
        )

        # Initialize empty dynamic context
        dynamic_context = {}

        # Main repair loop for multi-method fixing
        for iteration in range(5):
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Multi-Method Iteration {iteration + 1}/{self.config.max_iterations}")
            self.metrics['total_iterations'] += 1

            if monitor:
                monitor.log_iteration_start(iteration + 1)

            # Update dynamic context after first failure
            if iteration > 0:
                self.logger.info("Requesting LLM decision for multi-method context update...")

                # Combine all buggy method codes for context decision
                combined_buggy_code = "\n\n".join([
                    f"// Method: {m.method_name}\n{m.buggy_code if hasattr(m, 'buggy_code') else ''}"
                    for m in buggy_methods
                ])

                # Get LLM decision on what to retrieve
                decision = self.context_manager.decide_context_update(
                    buggy_method=combined_buggy_code,
                    kb=knowledge_base,
                    tried_hypotheses=self.hypothesis_pool.get_recent_hypotheses(),
                    current_context=dynamic_context,
                    retrieval_history=self.context_manager.state.retrieval_history
                )

                if monitor:
                    monitor.log_context_decision(decision)

                self.logger.info(f"LLM decided to use tools: {[cmd['tool'] for cmd in decision.get('tool_commands', [])]}")

                # Execute LLM decisions with validated commands
                dynamic_context = self.context_manager.execute_context_decisions(
                    current_context=dynamic_context,
                    decision=decision,
                    tools=self.tools,
                    bug_info={
                        'bug_id': bug_info.get('bug_id'),
                        'buggy_methods': [m.buggy_code for m in buggy_methods if hasattr(m, 'buggy_code')],
                        'buggy_method_names': [m.method_name for m in buggy_methods],
                        'buggy_classes': [m.class_name for m in buggy_methods],
                        'buggy_file_paths': list(set([m.file_path for m in buggy_methods])),
                        'failing_tests': failing_tests,
                        'project_path': self.executor.project_path
                    }
                )

                # Extract and log tool execution logs
                if monitor and '_tool_logs' in dynamic_context:
                    monitor.log_tool_executions(dynamic_context['_tool_logs'])

                # Remove metadata key before using context
                dynamic_context.pop('_tool_logs', None)

                self.logger.info(f"Dynamic context updated: {len(dynamic_context)} items")
            
            # Generate prompt for multi-method fixing
            prompt = self._create_multi_method_prompt(
                buggy_methods=buggy_methods,
                knowledge_base=knowledge_base,
                dynamic_context=dynamic_context,
                iteration=iteration + 1,
                recent_hypotheses=self.hypothesis_pool.get_recent_hypotheses() if iteration > 0 else [],
                sibling_fixes=sibling_fixes if iteration >= 1 else [],
                prohibited_symbols=self.hypothesis_pool.get_prohibited_symbols()
            )
            # Generate patches for ALL methods (returns {'patches': [...], 'usage': {...}})
            gen_result = self.generator.generate_multi_patch(prompt, buggy_methods, 2)
            patch_hypothesises = gen_result['patches']
            gen_usage = gen_result.get('usage', {})

            if monitor:
                monitor.log_generation(prompt, gen_usage)
            
            # Process and add metadata to each patch hypothesis
            for patch_hypothesis in patch_hypothesises:
                # Parse multi-method patch response first
                parsed_patches = self._parse_multi_method_patch(patch_hypothesis, buggy_methods)
                
                # Ensure parsed_patches has the expected structure
                if not isinstance(parsed_patches, dict):
                    parsed_patches = {'methods': parsed_patches}
                
                # Convert methods to a list if it's a dict to ensure order preservation
                # This handles the case where _parse_multi_method_patch returns a dict keyed by method names
                method_patches_list = []
                if 'methods' in parsed_patches:
                    if isinstance(parsed_patches['methods'], dict):
                        # If it's a dict, convert to ordered list
                        # Assuming the order in the dict matches the order of buggy_methods
                        method_patches_list = list(parsed_patches['methods'].values())
                    elif isinstance(parsed_patches['methods'], list):
                        method_patches_list = parsed_patches['methods']
                elif isinstance(parsed_patches, dict):
                    # If patches are at root level, extract them in order
                    for key in sorted(parsed_patches.keys()):
                        if key not in ['iteration', 'is_multi_method']:
                            method_patches_list.append(parsed_patches[key])
                
                # Add metadata to each method in the patch
                enriched_methods = {}
                for idx, method in enumerate(buggy_methods):
                    # Get line numbers for this method from knowledge base
                    line_numbers = knowledge_base.get('buggy_methods', [{}])[idx].get('lines', []) if idx < len(knowledge_base.get('buggy_methods', [])) else []
                    
                    # Create unique key for this method
                    line_identifier = f"L{line_numbers[0]}" if line_numbers else f"idx{idx}"
                    method_key = f"{method.file_path}::{method.class_name}::{method.method_name}::{line_identifier}"
                    
                    # Get the patch for this method by index (most reliable for duplicate names)
                    method_patch = None
                    if idx < len(method_patches_list):
                        method_patch = method_patches_list[idx]
                    else:
                        self.logger.warning(f"No patch found for method at index {idx}: {method.method_name}")
                    
                    # Create enriched method entry with all metadata
                    enriched_methods[method_key] = {
                        'file_path': method.file_path,
                        'method_name': method.method_name,
                        'class_name': method.class_name,
                        'line_numbers': line_numbers,
                        'original_code': method.buggy_code if hasattr(method, 'buggy_code') else '',
                        'fixed_method': method_patch['fixed_method'] if method_patch else '',
                        'iteration': iteration + 1
                    }
                
                # Update patch_hypothesis with enriched structure
                patch_hypothesis['methods'] = enriched_methods
                patch_hypothesis['iteration'] = iteration + 1
                patch_hypothesis['is_multi_method'] = True
                
                if monitor:
                    monitor.log_hypothesis(patch_hypothesis)
            
            # Execute patches
            best_exec_result_mm = None
            the_best_patch_for_insight = patch_hypothesises[0] if patch_hypothesises else None
            results = [{f'{iteration}-{idx}': None, 'patch': None} for idx in range(1, len(patch_hypothesises) + 1)]
            failed_patches_this_iter_mm = []  # collect all failures for pool feedback

            for idx, patch_hypothesis in enumerate(patch_hypothesises, 1):
                # Execute all patches together
                execution_result = self.executor.execute_multi_patches(
                    patch_hypothesis,
                    failing_tests
                )
                if monitor:
                    monitor.log_patch_result(patch_hypothesis, execution_result)

                results[idx - 1][f'{iteration}-{idx}'] = execution_result
                results[idx - 1]['patch'] = patch_hypothesis

                # Check result
                if execution_result['status'] == 'success':
                    self.logger.info(f"✓ Patch {idx} passes tests at iteration {iteration + 1}!")

                    # NEW: Check for overfitting if enabled
                    if self.config.enable_overfitting_detection:
                        is_overfitting, overfitting_analysis = self._detect_multi_method_overfitting(
                            patch_hypothesis, bug_info, failing_tests
                        )

                        if is_overfitting:
                            # Try to refine the overfitting patch
                            refined_patches = None
                            for refinement_attempt in range(1, self.config.max_refinement_attempts + 1):
                                refined_patches, fully_refined = self._refine_multi_method_overfitting(
                                    patch_hypothesis,
                                    overfitting_analysis,
                                    bug_info,
                                    failing_tests,
                                    buggy_methods,
                                    knowledge_base
                                )

                                if refined_patches and fully_refined:
                                    self.logger.info(f"✅ SUCCESSFULLY REFINED MULTI-METHOD PATCHES!")
                                    self.metrics["successful_patches"] += 1
                                    results[idx - 1]['patch'] = refined_patches
                                    break
                                elif refined_patches:
                                    # If we got partially refined patches, continue to next attempt
                                    results[idx - 1]['patch'] = refined_patches
                                    patch_hypothesis = refined_patches  # Update for next refinement attempt

                            # If refinement failed, continue to next patch candidate
                            if not (refined_patches and fully_refined):
                                self.logger.warning(f"Failed to refine overfitting multi-method patches after {self.config.max_refinement_attempts} attempts")
                                continue
                        else:
                            # Patches are semantically correct
                            self.logger.info(f"✅ SUCCESS with semantically correct patch {idx}!")
                            self.metrics["successful_patches"] += 1
                            if monitor:
                                monitor.finish_iteration()
                            return patch_hypothesis
                    else:
                        # Overfitting detection disabled, accept patches
                        self.logger.info(f"✅ SUCCESS with patch {idx} (semantic validation disabled)")
                        self.metrics["successful_patches"] += 1
                        if monitor:
                            monitor.finish_iteration()
                        return patch_hypothesis
                else:
                    self.logger.warning(f"✗ Failed (patch {idx}): {execution_result.get('error_type')}")
                    self.metrics["failed_patches"] += 1
                    # Bug 4 fix (multi): select best patch by fewest failing tests via execution_result
                    n_failing = len(execution_result.get('failed_tests', []))
                    n_best = len(best_exec_result_mm.get('failed_tests', [])) if best_exec_result_mm else float('inf')
                    if n_failing < n_best:
                        best_exec_result_mm = execution_result
                        the_best_patch_for_insight = patch_hypothesis
                        the_best_patch_for_insight['execution_result'] = execution_result
                    # Bug 2 fix: collect every failed patch for pool feedback
                    patch_hypothesis['execution_result'] = execution_result
                    failed_patches_this_iter_mm.append(patch_hypothesis)

            # Finish iteration tracking
            if monitor:
                monitor.finish_iteration()

            # Check if any patch succeeded
            plausible_patches = []
            for entry in results:
                result_key = next((k for k in entry.keys() if k != 'patch'), None)
                if result_key and entry[result_key] and entry[result_key]['status'] == 'success':
                    plausible_patches.append(entry['patch'])

            if len(plausible_patches) > 0:
                return plausible_patches
            else:
                # Bug 2+3 fix: add ALL unique failed patches to pool
                final_exec_mm = best_exec_result_mm if best_exec_result_mm else execution_result
                for failed_patch in failed_patches_this_iter_mm:
                    self.hypothesis_updater.analyze_and_update(
                        failed_patch,
                        failed_patch['execution_result'],
                        self.hypothesis_pool
                    )
                insights = self.hypothesis_updater.analyze_and_update(
                    the_best_patch_for_insight,
                    final_exec_mm,
                    self.hypothesis_pool
                )

            if self._should_stop_early(iteration, insights):
                self.logger.info("Stopping early based on insights")
                break

        self.logger.warning(
            f"Failed to fix {len(buggy_methods)} methods after {self.config.max_iterations} iterations "
            f"and {len(patch_hypothesises)} candidates"
        )
        return None

    def _detect_multi_method_overfitting(self, multi_patches: Dict, bug_info: Dict, 
                                        failing_tests: List[str]) -> Tuple[bool, Dict]:
        """Detect overfitting in multi-method patches."""
        self.logger.info("Checking multi-method patches for overfitting...")
        
        # Check each method for overfitting
        overfitting_detected = False
        all_analyses = []
        
        for method_patch in multi_patches.get('methods', []):
            method_patch = multi_patches['methods'][method_patch] 
            method_name = method_patch.get('method_name')
            original_code = method_patch.get('original_code','')
            
            single_patch = {
                'hypothesis': multi_patches.get('hypothesis'),
                'fixed_method': method_patch.get('fixed_method'),
                'original_code': original_code,
                'method_name': method_name,
                'changes': multi_patches.get('changes', '')
            }
            
            can_be_improved, is_overfit, analysis = self._detect_overfitting(single_patch, bug_info, failing_tests)
            all_analyses.append(analysis)
            
            if is_overfit:
                overfitting_detected = True
        
        # Combine analyses
        combined_analysis = {
            'is_overfitting': overfitting_detected,
            'method_analyses': all_analyses,
            'issues': [issue for a in all_analyses for issue in a.get('issues', [])],
            'refinement_suggestions': [s for a in all_analyses for s in a.get('refinement_suggestions', [])]
        }
        
        return overfitting_detected, combined_analysis
    
    def _refine_multi_method_overfitting(self, multi_patches: Dict, overfitting_analysis: Dict,
                                        bug_info: Dict, failing_tests: List[str],
                                        buggy_methods: List[FaultLocation], knowledge_base) -> Optional[Dict]:
        """Refine multi-method patches that show overfitting."""
        self.logger.info("Refining multi-method overfitting patches...")
        
        for attempt in range(1, self.config.max_refinement_attempts + 1):
            # Create refinement prompt for all methods
            prompt = self._create_multi_method_refinement_prompt(
                multi_patches, overfitting_analysis, buggy_methods, attempt
            )
            
            # Generate refined patches
            gen_result = self.generator.generate_multi_patch(prompt, buggy_methods, 3)
            refined_patches = gen_result['patches']

            for patch_hypothesis in refined_patches:
                                # Parse multi-method patch response first
                parsed_patches = self._parse_multi_method_patch(patch_hypothesis, buggy_methods)
                
                # Ensure parsed_patches has the expected structure
                if not isinstance(parsed_patches, dict):
                    parsed_patches = {'methods': parsed_patches}
                
                # Convert methods to a list if it's a dict to ensure order preservation
                # This handles the case where _parse_multi_method_patch returns a dict keyed by method names
                method_patches_list = []
                if 'methods' in parsed_patches:
                    if isinstance(parsed_patches['methods'], dict):
                        # If it's a dict, convert to ordered list
                        # Assuming the order in the dict matches the order of buggy_methods
                        method_patches_list = list(parsed_patches['methods'].values())
                    elif isinstance(parsed_patches['methods'], list):
                        method_patches_list = parsed_patches['methods']
                elif isinstance(parsed_patches, dict):
                    # If patches are at root level, extract them in order
                    for key in sorted(parsed_patches.keys()):
                        if key not in ['iteration', 'is_multi_method']:
                            method_patches_list.append(parsed_patches[key])
                
                # Add metadata to each method in the patch
                enriched_methods = {}
                for idx, method in enumerate(buggy_methods):
                    # Get line numbers for this method from knowledge base
                    line_numbers = knowledge_base.get('buggy_methods', [{}])[idx].get('lines', []) if idx < len(knowledge_base.get('buggy_methods', [])) else []
                    
                    # Create unique key for this method
                    line_identifier = f"L{line_numbers[0]}" if line_numbers else f"idx{idx}"
                    method_key = f"{method.file_path}::{method.class_name}::{method.method_name}::{line_identifier}"
                    
                    # Get the patch for this method by index (most reliable for duplicate names)
                    method_patch = None
                    if idx < len(method_patches_list):
                        method_patch = method_patches_list[idx]
                    else:
                        self.logger.warning(f"No patch found for method at index {idx}: {method.method_name}")
                    
                    # Create enriched method entry with all metadata
                    enriched_methods[method_key] = {
                        'file_path': method.file_path,
                        'method_name': method.method_name,
                        'class_name': method.class_name,
                        'line_numbers': line_numbers,
                        'original_code': method.buggy_code if hasattr(method, 'buggy_code') else '',
                        'fixed_method': method_patch['fixed_method'] if method_patch else '',
                        'attempt': attempt
                    }
                
                # Update patch_hypothesis with enriched structure
                patch_hypothesis['methods'] = enriched_methods
                patch_hypothesis['is_multi_method'] = True
                
                patch_hypothesis['is_refinement'] = True
                patch_hypothesis['refinement_attempt'] = attempt
                
                # Test refined patches
                execution_result = self.executor.execute_multi_patches(
                    patch_hypothesis,
                    failing_tests
                )
            
                if execution_result['status'] == 'success':
                    # Check if still overfitting
                    is_still_overfit, new_analysis = self._detect_multi_method_overfitting(
                        patch_hypothesis, bug_info, failing_tests
                    )
                    
                    if not is_still_overfit:
                        self.logger.info("✅ Multi-method refinement successful!")
                        self.metrics['successful_refinements'] += 1
                        return patch_hypothesis, True
                    else:
                        return patch_hypothesis, False
        self.metrics['failed_refinements'] += 1
        return multi_patches, False
    
    def _create_multi_method_refinement_prompt(self, multi_patches: Dict, 
                                              overfitting_analysis: Dict,
                                              buggy_methods: List[FaultLocation],
                                              attempt: int) -> str:
        """Create refinement prompt for multi-method patches."""
        prompt = f"""## Multi-Method Refinement - Attempt {attempt}

### CONTEXT:
The following {len(buggy_methods)} interdependent methods need refinement to fix overfitting.

### OVERFITTING ISSUES:
"""
        for issue in overfitting_analysis.get('issues', []):
            prompt += f"- {issue}\n"
        
        prompt += "\n### CURRENT PATCHES (OVERFITTING):\n"
        
        for method_patch in multi_patches.get('methods', []):
            prompt += f"""
Method: {multi_patches['methods'][method_patch].get('method_name')}
```java
{multi_patches['methods'][method_patch].get('fixed_method', '')}
```
"""
        
        prompt += f"""

### TASK:
Refine ALL methods to:
1. Fix root causes, not symptoms
2. Handle general cases
3. Work together correctly
4. Pass all tests without overfitting

Return JSON with fixes for ALL methods:
{{
    "hypothesis": "Overall explanation of the overfitment and coordinated refinement strategy",
    "methods": [
        {{
            "method_name": "method1_name",
            "fixed_method": "Complete fixed method1 code"
        }},
        {{
            "method_name": "method2_name", 
            "fixed_method": "Complete fixed method2 code"
        }}
        // ... fixes for all {len(buggy_methods)} methods
    ]
    "changes": "short apllied changes"
}}
"""
        return prompt
    
    def initialize_components(self, project_path: str, vector_db_path: str):
        """Initialize all system components with fixed imports."""
        from src.tools.fault_localization import FaultLocalizer
        from src.agents.generator_agents import PromptMakerAgent, GeneratorAgent
        from src.agents.hypothesis_manager import HypothesisPool, HypothesisUpdater
        from src.execution.executor import DockerExecutor, PatchExecutor
        from src.agents.llm_context_manager import LLMContextManager
        
        # Initialize fault localizer
        self.fault_localizer = FaultLocalizer(
            fl_tool=self.config.fl_tool,
            project_path=project_path
        )
        
        # Initialize agents
        self.prompt_maker = PromptMakerAgent()
        
        self.generator = GeneratorAgent(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=self.config.llm_model
        )
        
        # Initialize executor
        if self.config.use_docker:
            self.executor = DockerExecutor(project_path)
        else:
            self.executor = PatchExecutor(project_path)
        
        # Initialize hypothesis management
        self.hypothesis_pool = HypothesisPool(
            max_size=self.config.max_hypothesis_pool_size
        )
        
        self.hypothesis_updater = HypothesisUpdater()
        
        # Initialize LLM context manager with smart resolution
        self.context_manager = LLMContextManager(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=self.config.llm_model,
            max_tokens=self.config.knowledge_base_token_limit,
            project_path=project_path
        )
        
        # Initialize smart resolution if enabled
        if self.config.enable_smart_resolution:
            self.context_manager.initialize_smart_resolution(project_path)
        
        # Initialize tools
        self._initialize_tools(project_path, vector_db_path)
    
    def _initialize_tools(self, project_path: str, vector_db_path: str):
        """Initialize tool instances with enhanced CodeExtractor."""
        from src.tools.code_tools import (
            SimilarMethodSearcher,
            CodeExtractor,
            CallGraphBuilder,
            CoverageRunner,
            FieldDependencyAnalyzer,
            APIUsageFinder
        )
        
        # Initialize tools with project path
        self.tools = {
            'similar_method_search': SimilarMethodSearcher(
                index_path=f"{vector_db_path}/methods.index" if os.path.exists(f"{vector_db_path}/methods.index") else None,
                metadata_path=f"{vector_db_path}/methods_metadata.pkl" if os.path.exists(f"{vector_db_path}/methods_metadata.pkl") else None
            ),
            'code_extractor': CodeExtractor(project_path),  # Enhanced with test extraction
            'call_graph_builder': CallGraphBuilder(project_path),
            'coverage_runner': CoverageRunner(project_path, 'java'),
            'field_dependency_analyzer': FieldDependencyAnalyzer(project_path),
            'api_usage_finder': APIUsageFinder(project_path)
        }
        
        self.logger.info(f"Initialized {len(self.tools)} tools")
    
    def _load_perfect_fl_data(self):
        """Load perfect fault localization data from JSON file."""
        try:
            with open(self.config.perfect_fl_file, 'r') as f:
                self.perfect_fl_data = json.load(f)
            self.logger.info(f"Loaded perfect FL data for {len(self.perfect_fl_data)} bugs")
        except FileNotFoundError:
            self.logger.error(f"Perfect FL file not found: {self.config.perfect_fl_file}")
            self.perfect_fl_data = {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing perfect FL JSON: {e}")
            self.perfect_fl_data = {}
    
    def fix_bug(self, bug_info: Dict) -> Dict:
        """Main entry point for fixing a bug."""
        self.current_bug_info = bug_info  # Store for context
        bug_id = bug_info.get('bug_id')
        
        self.logger.info(f"Starting bug fix for {bug_id}")
        
        # Initialize monitor
        monitor = BugFixingMonitor(bug_id)
        self.monitors[bug_id] = monitor
        
        # Update context manager with bug info
        if self.context_manager:
            self.context_manager.update_bug_context(bug_info)
        
        # Run fault localization
        fault_locations = self._run_fault_localization(bug_info)
        
        if not fault_locations:
            self.logger.error(f"No fault locations found for {bug_id}")
            return {'success': False, 'error': 'No fault locations found'}
        
        # Get failing tests
        failing_tests = bug_info.get('failing_tests', [])
        
        # Find connected components
        components = self._find_connected_components(fault_locations, failing_tests)
        
        self.logger.info(f"Found {len(components)} connected components")
        
        # Process components
        patches = self._process_components(components, bug_info)

        # Validate patches: ALL components must have a successful patch
        success = len(patches) == len(components) and len(patches) > 0
        
        # Generate report
        monitor.generate_report()
        
        return {
            'success': success,
            'patches': patches,
            'bug_id': bug_id,
            'iterations': self.metrics.get('max_component_iterations', self.metrics['total_iterations']),
            'monitor_dir': f"traces/{bug_id}/"
        }
    
    def _process_components(self, components: List[List[Tuple]], bug_info: Dict) -> List[Dict]:
        """Process all components (can be single or multiple)."""
        patches = []

        if self.config.parallel_components and len(components) > 1:
            # Process in parallel — track per-component iterations via snapshots
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(components), 4)) as executor:
                futures = []
                for i, component in enumerate(components):
                    future = executor.submit(
                        self._process_component,
                        component,
                        bug_info,
                        i
                    )
                    futures.append(future)

                for future in futures:
                    result = future.result()
                    if result:
                        patches.append(result)
            # Parallel: approximate as total divided by component count
            n = max(len(components), 1)
            self.metrics['max_component_iterations'] = self.metrics['total_iterations'] // n
        else:
            # Sequential: track iterations used by each component individually
            sibling_fixes = []
            max_iters = 0
            for i, component in enumerate(components):
                self.logger.info(f"Processing component {i+1}/{len(components)}")
                iters_before = self.metrics['total_iterations']
                patch = self._process_component(component, bug_info, i, sibling_fixes=list(sibling_fixes))
                component_iters = self.metrics['total_iterations'] - iters_before
                max_iters = max(max_iters, component_iters)
                if patch:
                    patches.append(patch)
                    # _process_single_method may return a list (plausible_patches) or a
                    # single dict (early-return path). Normalize to a single representative
                    # patch dict before storing in sibling_fixes.
                    first_patch = patch[0] if isinstance(patch, list) else patch
                    if isinstance(first_patch, dict):
                        sibling_fixes.append(first_patch)
            self.metrics['max_component_iterations'] = max_iters

        return patches
    
    def _process_component(self, component: List[Tuple], bug_info: Dict,
                           component_index: int = 0,
                           sibling_fixes: List[Dict] = None) -> Optional[Dict]:
        """
        Process a component that can have MULTIPLE buggy methods and failing tests.
        All buggy methods in the component need to be fixed together.
        """
        if not component:
            return None

        # Extract ALL unique buggy methods and failing tests from component
        buggy_methods = component[0]
        failing_tests = component[1]

        self.logger.info(f"Processing component with {len(buggy_methods)} buggy methods and {len(failing_tests)} failing tests")
        for method in buggy_methods:
            self.logger.info(f"  - Buggy method: {method.method_name} in {method.file_path}")
        self.logger.info(f"  - Failing tests: {', '.join(failing_tests.keys())}")

        # Notify monitor of the current component context so iteration files go to the right place
        monitor = self.monitors.get(bug_info.get('bug_id'))
        if monitor:
            first = buggy_methods[0]
            monitor.set_component(
                index=component_index,
                method_name=first.method_name,
                file_path=first.file_path,
                class_name=first.class_name,
                failing_tests=list(failing_tests.keys())
            )

        # Determine if this is truly multi-method or single-method
        if len(buggy_methods) == 1:
            return self._process_single_method(buggy_methods[0], failing_tests, bug_info,
                                               sibling_fixes=sibling_fixes)
        else:
            return self._process_multi_method(buggy_methods, failing_tests, bug_info,
                                              sibling_fixes=sibling_fixes)
    
    def _create_single_method_prompt(self, buggy_method: FaultLocation, knowledge_base: Dict,
                                    dynamic_context: Dict, iteration: int,
                                    recent_hypotheses: List[Dict],
                                    sibling_fixes: List[Dict] = None,
                                    prohibited_symbols: List[str] = None) -> str:
        """Create prompt for single method fixing."""
        prompt = f"""## Bug Fixing - Iteration {iteration} (Single Method)

### BUGGY METHOD:
```java
{buggy_method.buggy_code if hasattr(buggy_method, 'buggy_code') else ''}
```

### FAILING TESTS ({len(knowledge_base['failing_tests'])} tests):
"""      
        # Add test information
        for i, test in enumerate(knowledge_base['failing_tests'], start=1):
            if i > 10:
                break
            test_key = f'test_code_{test}'
            if test_key in knowledge_base:
                test_info = knowledge_base[test_key]
                prompt += f"""
#### Test: {test}:
```java    
{test_info['code'] if 'code' in test_info else ''}
```
"""   
        # Add error messages
        if knowledge_base['error_messages']:
            prompt += "\n### ERROR MESSAGES:\n"
            for error in knowledge_base['error_messages'][:5]:
                prompt += f"""
- Test: {error['test']}
  Message: {error['error_message']}
"""
        
        # Add previous attempts
        if recent_hypotheses:
            prompt += "\n### PREVIOUSLY FAILED ATTEMPTS:\n"
            for i, hyp in enumerate(recent_hypotheses[-5:], 1):
                # Prefer the real unified diff; fall back to the LLM's free-text
                # description ('changes').  Use explicit None/empty check so that
                # an empty string "" does not silently skip to 'N/A'.
                diff_val    = hyp.get('diff')    or ''
                changes_val = hyp.get('changes') or ''
                code_diff   = diff_val if diff_val.strip() else (changes_val if changes_val.strip() else 'N/A')
                prompt += f"""
Attempt {i}:
 #### Applied Changes (unified diff):
```diff
{code_diff}
```
 #### Result: {hyp.get('failure_reason', 'unknown')}
"""
        # Add dynamic context
        if dynamic_context:
            prompt += "\n### RETRIEVED CONTEXT:\n"
            for key, value in list(dynamic_context.items())[:5]:
                prompt += f"\n#### {key.replace('_', ' ').title()}:\n"
                if isinstance(value, dict) and 'data' in value:
                    data = value['data']
                    # Format code results as readable text, not raw JSON
                    if isinstance(data, dict) and 'code' in data:
                        code_text = data['code']
                        if len(code_text) > 1500:
                            code_text = code_text[:1500] + '\n// ... truncated'
                        prompt += f"```java\n{code_text}\n```\n"
                    elif isinstance(data, list):
                        serialized = json.dumps(data, separators=(',', ':'))
                        if len(serialized) > 2000:
                            serialized = serialized[:2000] + '...]'
                        prompt += f"{serialized}\n"
                    else:
                        serialized = json.dumps(data, separators=(',', ':'))
                        if len(serialized) > 2000:
                            serialized = serialized[:2000] + '...}'
                        prompt += f"{serialized}\n"

        # Add sibling component fixes as context (from iteration 2 onward)
        if sibling_fixes:
            prompt += "\n### SIBLING METHODS ALREADY FIXED (same bug pattern — use as reference):\n"
            for fix in sibling_fixes:
                sib_method = fix.get('method_name', 'unknown')
                sib_class = fix.get('class_name', '')
                sib_diff = fix.get('diff', '')
                if sib_diff:
                    prompt += f"\n#### {sib_class}.{sib_method} — successful fix:\n"
                    prompt += f"```diff\n{sib_diff}\n```\n"

        if prohibited_symbols:
            prompt += "\n### PROHIBITED — These symbols do NOT exist in this codebase (do not use them):\n"
            for sym in prohibited_symbols:
                prompt += f"- {sym}\n"

        prompt += """
### TASK:
Fix the buggy method to make all failing tests pass.
"""
        return prompt

    def _create_multi_method_prompt(self, buggy_methods: List[FaultLocation], knowledge_base: Dict,
                                   dynamic_context: Dict, iteration: int,
                                   recent_hypotheses: List[Dict],
                                   sibling_fixes: List[Dict] = None,
                                   prohibited_symbols: List[str] = None) -> str:
        """Create prompt for multi-method fixing."""
        prompt = f"""## Bug Fixing - Iteration {iteration} (Multi-Method Component)

### These {len(buggy_methods)} methods are buggy methods that need to be fixed together.
### The corresponding failing tests are given below. 

### BUGGY METHODS ({len(buggy_methods)} methods to fix):
"""
        # Add all buggy methods
        for i, method in enumerate(buggy_methods, 1):
            prompt += f"""
#### Method {i}:
```java
{method.buggy_code if hasattr(method, 'buggy_code') else ''}
```
"""
        
        prompt += f"""
### FAILING TESTS ({len(knowledge_base['failing_tests'])} tests):
"""
        # Add test information
        for i, test in enumerate(knowledge_base['failing_tests'], start=1):
            if i > 5:
                break  # Limit to first 5 tests for brevity
            test_key = f'test_code_{test}'
            if test_key in knowledge_base:
                test_info = knowledge_base[test_key]
                prompt += f"""
#### Test: {test}
```java
{test_info['code'] if 'code' in test_info else ''}
```
"""
        
        # Add error messages
        if knowledge_base['error_messages']:
            prompt += "\n### ERROR MESSAGES:\n"
            for error in knowledge_base['error_messages'][:5]:
                prompt += f"""
- Test: {error['test']}
  Error: {error['error_type']} - {error['error_message'][:500]}
"""
        
        # Add previous attempts if any
        if recent_hypotheses:
            prompt += "\n### PREVIOUS MULTI-METHOD FIX ATTEMPTS:\n"
            for i, hyp in enumerate(recent_hypotheses[-5:], 1):
                # Explicit empty-string check so "" does not silently fall to 'N/A'
                diff_val    = hyp.get('diff')    or ''
                changes_val = hyp.get('changes') or ''
                code_diff   = diff_val if diff_val.strip() else (changes_val if changes_val.strip() else 'N/A')
                exec_res = hyp.get('execution_result') or {}
                # Build a meaningful result line: prefer stored failure_reason,
                # then compile-error details, then test lists.
                failure_reason = hyp.get('failure_reason', '')
                error_type     = exec_res.get('error_type', '')
                failed_tests   = exec_res.get('failed_tests', [])
                passed_tests   = exec_res.get('passed_tests', [])
                if failure_reason:
                    result_line = failure_reason
                elif error_type == 'CompilationError':
                    result_line = f"CompilationError — patch failed to compile. Do NOT call methods/variables that do not exist in the class."
                elif error_type:
                    result_line = f"{error_type}: failing tests = {failed_tests}"
                else:
                    result_line = f"Still failing: {failed_tests}" if failed_tests else "unknown"
                prompt += f"""
Attempt {i}:
- Hypothesis: {hyp.get('hypothesis', 'N/A')}
- Applied Changes (unified diff):
```diff
{code_diff}
```
- Result: {result_line}
    - Still failing tests: {failed_tests}
    - Passed tests: {passed_tests}
"""
        
        # Add dynamic context
        if dynamic_context:
            prompt += "\n### RETRIEVED CONTEXT:\n"
            for key, value in list(dynamic_context.items())[:5]:
                prompt += f"\n#### {key.replace('_', ' ').title()}:\n"
                if isinstance(value, dict) and 'data' in value:
                    data = value['data']
                    if isinstance(data, dict) and 'code' in data:
                        code_text = data['code']
                        if len(code_text) > 1500:
                            code_text = code_text[:1500] + '\n// ... truncated'
                        prompt += f"```java\n{code_text}\n```\n"
                    elif isinstance(data, list):
                        serialized = json.dumps(data, separators=(',', ':'))
                        if len(serialized) > 2000:
                            serialized = serialized[:2000] + '...]'
                        prompt += f"{serialized}\n"
                    else:
                        serialized = json.dumps(data, separators=(',', ':'))
                        if len(serialized) > 2000:
                            serialized = serialized[:2000] + '...}'
                        prompt += f"{serialized}\n"
        # Add sibling component fixes as context (from iteration 2 onward)
        if sibling_fixes:
            prompt += "\n### SIBLING METHODS ALREADY FIXED (same bug pattern — use as reference):\n"
            for fix in sibling_fixes:
                sib_method = fix.get('method_name', 'unknown')
                sib_class = fix.get('class_name', '')
                sib_diff = fix.get('diff', '')
                if sib_diff:
                    prompt += f"\n#### {sib_class}.{sib_method} — successful fix:\n"
                    prompt += f"```diff\n{sib_diff}\n```\n"

        if prohibited_symbols:
            prompt += "\n### PROHIBITED — These symbols do NOT exist in this codebase (do not use them):\n"
            for sym in prohibited_symbols:
                prompt += f"- {sym}\n"

        prompt += f"""

### TASK:
Provide fixes for ALL {len(buggy_methods)} methods that work together.
"""

        return prompt
    
    def _parse_multi_method_patch(self, patch_response: Dict, 
                                 buggy_methods: List[FaultLocation]) -> Dict:
        """Parse multi-method patch response from LLM."""
        if 'methods' in patch_response:
            # Already in multi-method format
            return {
                'hypothesis': patch_response.get('hypothesis', ''),
                'methods': patch_response['methods'],
                'is_multi_method': True,
                'changes': patch_response.get('changes', ''),
                'methods_count': len(buggy_methods)
            }
        else:
            # Single patch returned, need to map to multi-method format
            self.logger.warning("LLM returned single patch for multi-method component, adapting...")
            return {
                'hypothesis': patch_response.get('hypothesis', ''),
                'methods': [{
                    'method_name': buggy_methods[0].method_name,
                    'file_path': buggy_methods[0].file_path,
                    'changes': patch_response.get('changes', ''),
                    'fixed_method': patch_response.get('fixed_method', '')
                }],
                'is_multi_method': True,
                'incomplete': True,  # Mark as incomplete
                'methods_count': len(buggy_methods)
            }
    
    def _build_knowledge_base(self, buggy_methods: List[FaultLocation],
                             failing_tests: List[str], bug_info: Dict) -> Dict:
        """Build knowledge base for multiple buggy methods."""
        knowledge_base = {
            'buggy_methods': [],
            'failing_tests': failing_tests,
            'error_messages': []
        }
        
        # Add ALL buggy methods
        for method in buggy_methods:
            knowledge_base['buggy_methods'].append({
                'name': method.method_name,
                'code': method.buggy_code if hasattr(method, 'buggy_code') else '',
                'file': method.file_path,
                'lines': method.line_numbers,
                'class': method.class_name
            })
        
        # Extract test code using enhanced CodeExtractor
        for test in failing_tests:      
            rel_path = failing_tests[test].get('path', '')
            if bug_info.get('bug_id')in rel_path:
                rel_path = rel_path.replace(bug_info['bug_id'], '').lstrip('/')
            file_path = os.path.join(bug_info.get('project_path', ''), rel_path)
            test_result = self.tools['code_extractor'].extract(
                file_path=file_path,
                element_name=test,
                element_type='test'
            )
            
            if test_result and test_result.get('code'):
                knowledge_base[f'test_code_{test}'] = {
                    'code': test_result['code'],
                    'assertions': test_result.get('assertions', []),
                    'setup': test_result.get('setup', '')
                }
        
        # Get error messages
        knowledge_base['error_messages'] = self._get_error_messages(failing_tests, bug_info)
        
        return knowledge_base
    
    def _get_error_messages(self, failing_tests: List[str], bug_info: Dict) -> List[Dict]:
        """Get error messages for failing tests."""
        error_messages = []
        
        if 'failing_tests' in bug_info:
            for test in failing_tests:
                error_messages.append({
                    'test': test,
                    'error_type': self._extract_error_type(bug_info['failing_tests'][test].get('error_msg', '')),
                    'error_message': bug_info['failing_tests'][test].get('clean_error_msg', ''),
                    'stack_trace': bug_info['failing_tests'][test].get('error_msg', '')[:500]
                })
        else:
            for test in failing_tests:
                try:
                    result = self._run_single_test_for_error(test, 'java')
                    error_messages.append({
                        'test': test,
                        'error_type': result.get('error_type', 'Unknown'),
                        'error_message': result.get('error_message', ''),
                        'stack_trace': result.get('stack_trace', '')[:500]
                    })
                except Exception as e:
                    self.logger.error(f"Failed to get error for {test}: {e}")
        
        return error_messages
    
    def _run_fault_localization(self, bug_info: Dict) -> List[FaultLocation]:
        """Run fault localization to identify suspicious code locations."""
        self.logger.info(f"Running fault localization with {self.config.fl_tool}")
        
        if self.config.fl_tool == "perfect":
            return self._get_perfect_fault_locations(bug_info)
        else:
            failing_tests = bug_info.get('failing_tests', [])
            fault_locations = self.fault_localizer.run_fl_tool(failing_tests)
            return fault_locations[:10]
    
    def _get_perfect_fault_locations(self, bug_info: Dict) -> List[FaultLocation]:
        """Get perfect fault locations from JSON data."""
        bug_id = bug_info.get('bug_id')
        if not bug_id or bug_id not in self.perfect_fl_data:
            self.logger.warning(f"No perfect FL data found for bug {bug_id}")
            return []
        
        bug_data = self.perfect_fl_data[bug_id]
        perfect_locations = []
        
        # Handle both single and multi-function formats
        if 'function_num' in bug_data:
            # Multi-function format
            for func_info in bug_data.get('functions', []):
                location = FaultLocation(
                    file_path=func_info.get('path'),
                    class_name=self._extract_class_name(func_info.get('path')),
                    method_name=self._extract_method_name(func_info.get('buggy_function')),
                    suspiciousness_score=1.0,
                    line_numbers=[
                        func_info.get('start_loc', 0),
                        func_info.get('end_loc', 0)
                    ]
                )
                location.buggy_code = func_info.get('comment') + '\n' + func_info.get('buggy_function')
                location.fixed_code = func_info.get('fixed_function')
                perfect_locations.append(location)
        else:
            # Single function format
            location = FaultLocation(
                file_path=bug_data.get('loc'),
                class_name=self._extract_class_name(bug_data.get('loc')),
                method_name=self._extract_method_name_from_signature(bug_data.get('method_signature', {})),
                suspiciousness_score=1.0,
                line_numbers=list(range(
                    bug_data.get('start', 0),
                    bug_data.get('end', 0) + 1
                ))
            )
            location.buggy_code = bug_data.get('buggy_code_comment') + '\n' + bug_data.get('buggy')
            location.fixed_code = bug_data.get('fix')
            perfect_locations.append(location)
        
        # Store failing test details if available
        if 'failing_tests_details' in bug_data:
            bug_info['failing_tests_details'] = bug_data['failing_tests_details']
        
        return perfect_locations
    
    def extract_method_info(self, java_code: str):
        import re

        # -------- Step 1: Remove comments safely (preserve string literals) --------
        comment_pattern = re.compile(
            r"""
            ("(?:\\.|[^"\\])*")     |  # Double quoted strings
            ('(?:\\.|[^'\\])*')     |  # Single quoted strings
            (//.*?$)                |  # Line comments
            (/\*.*?\*/)                # Block / JavaDoc comments
            """,
            re.MULTILINE | re.DOTALL | re.VERBOSE,
        )

        def comment_replacer(match):
            # If group 3 or 4 matched → it's a comment
            if match.group(3) or match.group(4):
                return ""
            return match.group(0)

        clean_code = re.sub(comment_pattern, comment_replacer, java_code)

        # -------- Step 2: Extract method signature --------
        method_pattern = re.compile(
            r"""
            (?:@\w+(?:\([^)]*\))?\s+)*       # Annotations
            (?:public|private|protected)?\s* # Access modifier
            (?:static\s+)?                   # Optional static
            (?:final\s+)?                    # Optional final
            (?:synchronized\s+)?             # Optional synchronized
            (?:[\w<>\[\],?]+\s+)+             # Return type
            (?P<name>\w+)                    # Method name
            \s*\(
            (?P<params>[^)]*)
            \)
            """,
            re.VERBOSE | re.MULTILINE | re.DOTALL
        )

        match = method_pattern.search(clean_code)
        if not match:
            return None

        method_name = match.group("name")
        params = match.group("params").strip()

        if not params:
            return {
                "name": method_name,
                "num_params": 0,
                "param_types": []
            }

        # -------- Step 3: Robust parameter splitting (handles generics) --------
        raw_params = [
            p.strip()
            for p in re.split(r",(?![^<]*>)", params)
            if p.strip()
        ]

        param_types = []
        for p in raw_params:
            parts = p.split()
            if not parts:
                continue

            # All but last token → type
            if len(parts) > 1:
                ptype = " ".join(parts[:-1])
            else:
                ptype = parts[0]

            param_types.append(ptype.replace("...", "[]"))  # varargs → array

        return {
            "name": method_name,
            "num_params": len(param_types),
            "param_types": param_types
        }

    def _find_connected_components(self, fault_locations: List[FaultLocation], 
                                  failing_tests: List[str]) -> List[List[Tuple]]:
        """
        Find connected components of fault locations and tests.
        A component contains all buggy methods and tests that are related.
        """
        error_components = ['Closure-156', 'Closure-167', 'Closure-171', 'Closure-174', 'Compress-4', 'Gson-4', 'Gson-14', 'Jsoup-83', 'Math-35', 'Math-46', 'Math-66', 'Mockito-6']
        components = []
        if len(fault_locations) == 1 or len(failing_tests) == 1:
            # Single fault location - all tests connect to it
            components.append([fault_locations, failing_tests])
            return components

        p = os.path.join(ready_components_path, f"{self.current_bug_info.get('bug_id', '')}/grouped_by_overlap.json")

        with open(p, 'r') as f:
            grouped_components = json.load(f)
        
        
        components = []
        for group in grouped_components:
            component_fls = []
            component_tests = {}
             
            group_methods = [] 
            for method in grouped_components[group]['buggy_methods']:
                method_name = method.split('_')[0]
                class_name = method.split('_')[-1].split('.')[-1]
                type = ""
                if len(method.split('_'))==3:
                    type = method.split('_')[1]
                else:
                    type = '-'
                group_methods.append([method_name, class_name, type])
            
            group_tests = [m for m in grouped_components[group]['tests']]

            for fl in fault_locations:
                types = "".join(self.extract_method_info(fl.buggy_code)['param_types'])
                for g in group_methods:
                    if g[2] == '-':
                        if [fl.method_name, fl.class_name, '-'] in group_methods:
                            if fl not in component_fls:
                                component_fls.append(fl)
                    else:
                        if [fl.method_name, fl.class_name, types] in group_methods:
                            if fl not in component_fls:
                                component_fls.append(fl)
            for test in failing_tests:
                if test in group_tests:
                    component_tests[test] = failing_tests[test]
            
           

            if component_fls and component_tests:
                components.append([component_fls, component_tests])
        total_fls = []
        for cm in components:
            total_fls += cm[0]
        if len(fault_locations)!=len(total_fls):
            remaining_fls = [fl for fl in fault_locations if fl not in total_fls]
            for cm in components:
                cm[0]+=remaining_fls
        return components
    
    def _should_stop_early(self, iteration: int, insights: Dict) -> bool:
        """Determine if should stop early based on insights."""
        # Stop if no progress after many iterations
        if iteration > 10 and insights.get('no_progress'):
            return True
     
    def _extract_error_type(self, error_msg: str) -> str:
        """Extract error type from error message."""
        error_types = [
            'NullPointerException',
            'AssertionError',
            'IndexOutOfBoundsException',
            'ClassCastException',
            'IllegalArgumentException',
            'CompilationError'
        ]
        
        for error_type in error_types:
            if error_type in error_msg:
                return error_type
        
        return 'Unknown'
    
    def _extract_class_name(self, file_path: str) -> str:
        """Extract class name from file path."""
        if not file_path:
            return ''
        
        filename = os.path.basename(file_path)
        if '.' in filename:
            return filename.rsplit('.', 1)[0]
        return filename
    
    def _extract_method_name(self, method_code: str) -> str:
        """Extract method name from Java method code."""
        pattern = re.compile(
            r"(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|\s)*"  # modifiers
            r"[\w<>\[\],\s]+\s+"  # return type (with generics/arrays)
            r"(?P<name>\w+)\s*\("  # method name before '('
        )

        match = pattern.search(method_code)
        if match:
            return match.group("name")
        return "unknown"
    
    def _extract_method_name_from_signature(self, signature: Dict) -> str:
        """Extract method name from signature dictionary."""
        if isinstance(signature, dict):
            return signature.get('method_name', 'unknown')
        return 'unknown'


class BugFixingMonitor:
    """
    Comprehensive monitor that tracks complete bug fixing progress.
    Saves traces in a structured per-bug directory:
      traces/<bug_id>/summary.json, iterations/iteration_N.json,
      tools/tool_executions.json, cost.json
    """

    # GPT-4o pricing (per token)
    PRICING = {
        'model': 'gpt-4o',
        'input_per_1k': 0.0025,
        'output_per_1k': 0.01
    }

    def __init__(self, bug_id: str, output_dir: str = "traces"):
        self.bug_id = bug_id
        self.base_output_dir = output_dir
        self.output_dir = os.path.join(output_dir, bug_id)
        self.start_time = time.time()

        # Per-iteration structured data
        self.iterations = []  # List of complete iteration dicts
        self.current_iteration = {}

        # Flat logs for cross-cutting concerns
        self.all_tool_executions = []
        self.all_api_calls = []  # Each: {type, iteration, prompt_tokens, completion_tokens, cost_usd}
        self.error_patterns = {}
        self.hypothesis_evolution = []

        # Component tracking (for multi-component bugs like Chart-14)
        self.current_component_index = None
        self.current_component_info = {}

        # Create output directories
        os.makedirs(os.path.join(self.output_dir, 'iterations'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'tools'), exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(f"Monitor_{bug_id}")
        handler = logging.FileHandler(os.path.join(self.output_dir, 'monitor.log'))
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"Monitor initialized for bug: {bug_id}")

    def set_component(self, index: int, method_name: str = None, file_path: str = None,
                      class_name: str = None, failing_tests: list = None):
        """Set the current component context for multi-component bugs."""
        self.current_component_index = index
        self.current_component_info = {
            'component_index': index,
            'method_name': method_name,
            'file_path': file_path,
            'class_name': class_name,
            'failing_tests': list(failing_tests) if failing_tests else []
        }
        # Save component_info.json in a per-component subdir
        component_dir = os.path.join(self.output_dir, 'iterations', f'component_{index}')
        os.makedirs(component_dir, exist_ok=True)
        info_file = os.path.join(component_dir, 'component_info.json')
        with open(info_file, 'w') as f:
            json.dump(self.current_component_info, f, indent=2)
        self.logger.info(
            f"{'='*60}\nComponent {index}: {class_name}.{method_name} ({file_path})\n{'='*60}"
        )

    def log_iteration_start(self, iteration: int):
        """Start tracking a new iteration."""
        self.current_iteration = {
            'iteration': iteration,
            'component_index': self.current_component_index,
            'component_method': self.current_component_info.get('method_name'),
            'component_class': self.current_component_info.get('class_name'),
            'component_file': self.current_component_info.get('file_path'),
            'start_time': time.time(),
            'timestamp': datetime.now().isoformat(),
            'context_decision': None,
            'tool_executions': [],
            'generation': None,
            'patches': [],
            'duration_sec': 0
        }
        component_label = (
            f"Component {self.current_component_index} ({self.current_component_info.get('method_name', '?')})"
            if self.current_component_index is not None else ""
        )
        self.logger.info(f"{'='*60}\nStarting Iteration {iteration}  {component_label}\n{'='*60}")

    def log_context_decision(self, decision: Dict):
        """Log the context decision LLM call with prompt, response, and usage."""
        meta = decision.get('_meta', {})
        usage = meta.get('usage', {})

        self.current_iteration['context_decision'] = {
            'prompt': meta.get('prompt', ''),
            'response': meta.get('raw_response', ''),
            'parsed_decision': {k: v for k, v in decision.items() if k != '_meta'},
            'usage': usage
        }

        # Track API cost
        if usage:
            self._track_api_call('context_decision', usage)

        self.logger.debug(f"Logged context decision: {len(decision.get('tool_commands', []))} tools, usage={usage}")

    def log_tool_executions(self, tool_logs: list):
        """Log tool executions from execute_context_decisions."""
        for log_entry in tool_logs:
            # Serialize result safely
            result = log_entry.get('result')
            try:
                serialized = result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else str(result)
            except:
                serialized = "Could not serialize"

            entry = {
                'tool': log_entry.get('tool', ''),
                'params': log_entry.get('params', {}),
                'result': serialized,
                'status': log_entry.get('status', ''),
                'error': log_entry.get('error', ''),
                'execution_time_sec': log_entry.get('execution_time_sec', 0),
                'iteration': self.current_iteration.get('iteration', 0),
                'timestamp': time.time()
            }

            self.current_iteration['tool_executions'].append(entry)
            self.all_tool_executions.append(entry)

        self.logger.debug(f"Logged {len(tool_logs)} tool executions")

    def log_generation(self, prompt: str, usage: Dict):
        """Log the patch generation LLM call."""
        self.current_iteration['generation'] = {
            'prompt': prompt,
            'usage': usage
        }

        if usage:
            self._track_api_call('generation', usage)

        self.logger.debug(f"Logged generation: usage={usage}")

    def log_patch_result(self, patch: Dict, execution_result: Dict):
        """Log a single patch attempt and its execution result."""
        patch_entry = {
            'hypothesis': patch.get('hypothesis', ''),
            'changes': patch.get('changes', ''),
            'fixed_method': patch.get('fixed_method', ''),
            'diff': patch.get('diff', ''),
            'is_multi_method': patch.get('is_multi_method', False),
            'methods': patch.get('methods', {}),
            # Component identity fields (set on patch_hypothesis in _process_single/multi_method)
            'file_path': patch.get('file_path'),
            'method_name': patch.get('method_name'),
            'class_name': patch.get('class_name'),
            'iteration': patch.get('iteration', self.current_iteration.get('iteration')),
            'component_index': self.current_component_index,
            'execution_result': {
                'status': execution_result.get('status', ''),
                'error_type': execution_result.get('error_type', ''),
                'error_message': execution_result.get('error_message', ''),
                'failed_tests': execution_result.get('failed_tests', []),
                'passed_tests': execution_result.get('passed_tests', []),
                'compilation_success': execution_result.get('status') != 'compilation_error'
            }
        }

        self.current_iteration['patches'].append(patch_entry)

        # Track error patterns
        error_type = execution_result.get('error_type', 'Unknown')
        if execution_result.get('status') != 'success':
            self.error_patterns[error_type] = self.error_patterns.get(error_type, 0) + 1

        self.logger.info(f"Patch result: {execution_result.get('status')} ({error_type})")

    def log_refinement(self, refinement_data: Dict):
        """Log overfitting detection and refinement attempts."""
        if 'refinements' not in self.current_iteration:
            self.current_iteration['refinements'] = []
        self.current_iteration['refinements'].append(refinement_data)

    def finish_iteration(self):
        """Finalize current iteration and save to file."""
        self.current_iteration['duration_sec'] = time.time() - self.current_iteration.get('start_time', time.time())

        # Save iteration file under a component-specific subdir to avoid overwrites
        iteration_num = self.current_iteration.get('iteration', len(self.iterations) + 1)
        if self.current_component_index is not None:
            component_dir = os.path.join(self.output_dir, 'iterations', f'component_{self.current_component_index}')
            os.makedirs(component_dir, exist_ok=True)
            iteration_file = os.path.join(component_dir, f'iteration_{iteration_num}.json')
        else:
            iteration_file = os.path.join(self.output_dir, 'iterations', f'iteration_{iteration_num}.json')
        try:
            with open(iteration_file, 'w') as f:
                json.dump(self.current_iteration, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save iteration file: {e}")

        self.iterations.append(self.current_iteration)
        self.logger.info(f"Iteration {iteration_num} saved ({self.current_iteration['duration_sec']:.1f}s)")

    # --- Legacy compatibility methods ---

    def log_prompt(self, prompt: str, prompt_type: str = "generation", metadata: Dict = None):
        """Legacy: log a prompt (now handled by log_generation/log_context_decision)."""
        pass

    def log_llm_response(self, response: str, response_type: str = "patch", metadata: Dict = None):
        """Legacy: log LLM response (now handled by log_context_decision/log_generation)."""
        pass

    def log_context_update(self, context: Dict, new_info: Dict, tool_used: str = None):
        """Legacy: context update logging (now handled by log_tool_executions)."""
        pass

    def log_hypothesis(self, hypothesis: Dict):
        """Log hypothesis for tracking."""
        self.hypothesis_evolution.append({
            'iteration': self.current_iteration.get('iteration', 0),
            'hypothesis': hypothesis.get('hypothesis', ''),
            'changes': hypothesis.get('changes', ''),
            'fixed_method': hypothesis.get('fixed_method', ''),
            'is_multi_method': hypothesis.get('is_multi_method', False),
            'timestamp': time.time()
        })

    def log_execution_result(self, result: Dict):
        """Legacy: now handled by log_patch_result."""
        pass

    def log_iteration_start_legacy(self, iteration: int):
        """Legacy compatibility."""
        self.log_iteration_start(iteration)

    # --- API cost tracking ---

    def _track_api_call(self, call_type: str, usage: Dict):
        """Track an API call with its cost."""
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        cost = (prompt_tokens / 1000 * self.PRICING['input_per_1k'] +
                completion_tokens / 1000 * self.PRICING['output_per_1k'])

        self.all_api_calls.append({
            'type': call_type,
            'iteration': self.current_iteration.get('iteration', 0),
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'cost_usd': round(cost, 6)
        })

    # --- Report generation ---

    def generate_report(self):
        """Generate and save the complete structured trace output."""
        # 1. Save summary.json
        summary = self._create_summary()
        self._save_json(os.path.join(self.output_dir, 'summary.json'), summary)

        # 2. Save tools/tool_executions.json
        self._save_json(os.path.join(self.output_dir, 'tools', 'tool_executions.json'),
                       self.all_tool_executions)

        # 3. Save cost.json
        cost = self._create_cost_report()
        self._save_json(os.path.join(self.output_dir, 'cost.json'), cost)

        self.logger.info(f"Complete report saved to {self.output_dir}/")
        return summary

    def _create_summary(self) -> Dict:
        """Create the top-level summary."""
        total_patches = sum(len(it.get('patches', [])) for it in self.iterations)
        compiled_patches = sum(
            1 for it in self.iterations
            for p in it.get('patches', [])
            if p.get('execution_result', {}).get('compilation_success', False)
        )
        passed_patches = sum(
            1 for it in self.iterations
            for p in it.get('patches', [])
            if p.get('execution_result', {}).get('status') == 'success'
        )

        tools_used = {}
        for t in self.all_tool_executions:
            name = t.get('tool', 'unknown')
            tools_used[name] = tools_used.get(name, 0) + 1

        total_prompt_tokens = sum(c.get('prompt_tokens', 0) for c in self.all_api_calls)
        total_completion_tokens = sum(c.get('completion_tokens', 0) for c in self.all_api_calls)
        total_cost = sum(c.get('cost_usd', 0) for c in self.all_api_calls)

        return {
            'bug_id': self.bug_id,
            'status': 'success' if passed_patches > 0 else 'failed',
            'total_time_sec': round(time.time() - self.start_time, 2),
            'total_iterations': len(self.iterations),
            'total_patches_generated': total_patches,
            'total_patches_compiled': compiled_patches,
            'total_patches_passed': passed_patches,
            'error_distribution': self.error_patterns,
            'tools_used': tools_used,
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total': total_prompt_tokens + total_completion_tokens
            },
            'estimated_cost_usd': round(total_cost, 4),
            'api_calls_count': len(self.all_api_calls)
        }

    def _create_cost_report(self) -> Dict:
        """Create the detailed cost breakdown."""
        total_prompt = sum(c.get('prompt_tokens', 0) for c in self.all_api_calls)
        total_completion = sum(c.get('completion_tokens', 0) for c in self.all_api_calls)
        total_cost = sum(c.get('cost_usd', 0) for c in self.all_api_calls)

        return {
            'model': self.PRICING['model'],
            'pricing': {
                'input_per_1k': self.PRICING['input_per_1k'],
                'output_per_1k': self.PRICING['output_per_1k']
            },
            'calls': self.all_api_calls,
            'totals': {
                'prompt_tokens': total_prompt,
                'completion_tokens': total_completion,
                'total_tokens': total_prompt + total_completion,
                'total_cost_usd': round(total_cost, 4)
            }
        }

    def _save_json(self, path: str, data):
        """Save data as JSON file."""
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save {path}: {e}")