# Enhanced controller.py with overfitting detection and refinement
"""
Enhanced controller that detects overfitting and refines patches.
Adds semantic validation after test passing to prevent overfitting.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import json
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import os
from datetime import datetime
import re

from src.tools.fault_localization import FaultLocation

ready_components_path = 'data/connected_components'

@dataclass
class BugFixingConfig:
    max_iterations: int = 10
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
            response = self.context_manager.client.chat.completions.create(
                model=self.config.semantic_validation_model,
                messages=[
                    {"role": "system", "content": "You are an expert at detecting overfitting in program repair patches."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
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
            # On error, assume not overfitting to avoid blocking
            return False, {"error": str(e)}
    
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
            refined_patches = self.generator.generate_patch(prompt, patch.get('fixed_method', ''), 3)
            
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
                               failing_tests: List[str], bug_info: Dict) -> Optional[Dict]:
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
                    monitor.log_llm_response(
                        response=json.dumps(decision),
                        response_type="context_decision",
                        metadata={
                            'tools_selected': [cmd['tool'] for cmd in decision.get('tool_commands', [])],
                            'reasoning_length': len(decision.get('reasoning', '')),
                            'focus_area': decision.get('focus', '')
                        }
                    )
                
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
                
                if monitor:
                    monitor.log_context_update(
                        dynamic_context,
                        dynamic_context,
                        ', '.join([cmd['tool'] for cmd in decision.get('tool_commands', [])])
                    )
                
                self.logger.info(f"Dynamic context updated: {len(dynamic_context)} items")
            
            # Generate prompt for single method
            prompt = self._create_single_method_prompt(
                buggy_method=buggy_method,
                knowledge_base=knowledge_base,
                dynamic_context=dynamic_context,
                iteration=iteration + 1,
                recent_hypotheses=self.hypothesis_pool.get_recent_hypotheses() if iteration > 0 else []
            )
            print(prompt)
            if monitor:
                monitor.log_prompt(
                    prompt=prompt,
                    prompt_type="generation",
                    metadata={
                        'iteration': iteration + 1,
                        'method_name': buggy_method.method_name,
                        'failing_tests_count': len(failing_tests),
                        'context_size': len(dynamic_context),
                        'hypothesis_count': len(self.hypothesis_pool.get_recent_hypotheses() if iteration > 0 else [])
                    }
                )
        
            # Generate patch
            patch_hypothesises = self.generator.generate_patch(prompt, buggy_method.buggy_code if hasattr(buggy_method, 'buggy_code') else '', 10)
            
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
            the_best_patch_for_insight = patch_hypothesises[0] if patch_hypothesises else None
            results = [{f'{iteration}-{idx}': None, 'patch': None} for idx in range(1, len(patch_hypothesises) + 1)]
            for idx, patch_hypothesis in enumerate(patch_hypothesises, 1):
                execution_result = self.executor.execute_patch(
                    patch_hypothesis,
                    failing_tests
                )
                
                if monitor:
                    monitor.log_execution_result(execution_result)
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
                            # Try to refine the overfitting patch
                            refined_patch = None
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
                                    break
                                elif refined_patch:
                                    # If we got a partially refined patch, continue to next attempt
                                    results[idx - 1]['patch'] = refined_patch
                            
                            # If refinement failed, continue to next patch candidate
                            self.logger.warning(f"Failed to refine overfitting patch after {self.config.max_refinement_attempts} attempts")
                            continue
                        else:
                            # Patch is semantically correct
                            self.logger.info(f"✅ SUCCESS with semantically correct patch {idx}!")
                            self.metrics["successful_patches"] += 1
                    else:
                        # Overfitting detection disabled, accept patch
                        self.logger.info(f"✅ SUCCESS with patch {idx} (semantic validation disabled)")
                        self.metrics["successful_patches"] += 1
                        return patch_hypothesis
                else:
                    self.logger.warning(f"✗ Failed (patch {idx}): {execution_result.get('error_type')}")
                    self.metrics["failed_patches"] += 1
                    if len(execution_result.get('failing_tests',[]))<len(the_best_patch_for_insight.get('failing_tests', [])):
                        the_best_patch_for_insight = patch_hypothesis
            # Check if any patch succeeded
            plausible_patches = []
            for entry in results:
                if entry[next(iter(entry))]['status']=='success':
                    plausible_patches.append(entry['patch'])
            if len(plausible_patches)>0:
                return plausible_patches
            
            else:
                # Update hypothesis pool
                insights = self.hypothesis_updater.analyze_and_update(
                    the_best_patch_for_insight,
                    execution_result,
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
                         failing_tests: List[str], bug_info: Dict) -> Optional[Dict]:
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
                    monitor.log_llm_response(
                        response=json.dumps(decision),
                        response_type="context_decision",
                        metadata={
                            'tools_selected': [cmd['tool'] for cmd in decision.get('tool_commands', [])],
                            'reasoning_length': len(decision.get('reasoning', '')),
                            'focus_area': decision.get('focus', ''),
                            'methods_count': len(buggy_methods)
                        }
                    )
                
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
                
                if monitor:
                    monitor.log_context_update(
                        dynamic_context,
                        dynamic_context,
                        ', '.join([cmd['tool'] for cmd in decision.get('tool_commands', [])])
                    )
                
                self.logger.info(f"Dynamic context updated: {len(dynamic_context)} items")
            
            # Generate prompt for multi-method fixing
            prompt = self._create_multi_method_prompt(
                buggy_methods=buggy_methods,
                knowledge_base=knowledge_base,
                dynamic_context=dynamic_context,
                iteration=iteration + 1,
                recent_hypotheses=self.hypothesis_pool.get_recent_hypotheses() if iteration > 0 else []
            )
            if monitor:
                monitor.log_prompt(
                    prompt=prompt,
                    prompt_type="generation",
                    metadata={
                        'iteration': iteration + 1,
                        'method_names': [m.method_name for m in buggy_methods],
                        'methods_count': len(buggy_methods),
                        'failing_tests_count': len(failing_tests),
                        'context_size': len(dynamic_context),
                        'hypothesis_count': len(self.hypothesis_pool.get_recent_hypotheses() if iteration > 0 else [])
                    }
                )
            
            # Generate patches for ALL methods
            patch_hypothesises = self.generator.generate_multi_patch(prompt, buggy_methods, 10)
            
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
            the_best_patch_for_insight = patch_hypothesises[0] if patch_hypothesises else None
            results = [{f'{iteration}-{idx}': None, 'patch': None} for idx in range(1, len(patch_hypothesises) + 1)]
            
            for idx, patch_hypothesis in enumerate(patch_hypothesises, 1):
                # Execute all patches together
                execution_result = self.executor.execute_multi_patches(
                    patch_hypothesis,
                    failing_tests
                )
                if(idx==1):
                    the_best_patch_for_insight['execution_result'] = execution_result
                if monitor:
                    monitor.log_execution_result(execution_result)
                
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
                            return patch_hypothesis
                    else:
                        # Overfitting detection disabled, accept patches
                        self.logger.info(f"✅ SUCCESS with patch {idx} (semantic validation disabled)")
                        self.metrics["successful_patches"] += 1
                        return patch_hypothesis
                else:
                    self.logger.warning(f"✗ Failed (patch {idx}): {execution_result.get('error_type')}")
                    self.metrics["failed_patches"] += 1
                    if 'failed_tests' in the_best_patch_for_insight['execution_result']:
                        if len(execution_result.get('failed_tests', [])) < len(the_best_patch_for_insight['execution_result']['failed_tests']):
                            the_best_patch_for_insight = patch_hypothesis
                            the_best_patch_for_insight['execution_result'] = execution_result
            
            # Check if any patch succeeded
            plausible_patches = []
            for entry in results:
                result_key = next((k for k in entry.keys() if k != 'patch'), None)
                if result_key and entry[result_key] and entry[result_key]['status'] == 'success':
                    plausible_patches.append(entry['patch'])
            
            if len(plausible_patches) > 0:
                return plausible_patches
            else:
                # Update hypothesis pool
                insights = self.hypothesis_updater.analyze_and_update(
                    the_best_patch_for_insight,
                    the_best_patch_for_insight['execution_result'],  # This will be the last execution_result from the loop
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
    # def _process_multi_method(self, buggy_methods: List[FaultLocation], 
    #                          failing_tests: List[str], bug_info: Dict) -> Optional[Dict]:
    #     """Enhanced multi-method processing with overfitting detection."""
    #     self.logger.info(f"Processing {len(buggy_methods)} interdependent buggy methods")
        
    #     # Get monitor
    #     monitor = self.monitors.get(bug_info.get('bug_id'))
        
    #     # Build knowledge base for ALL methods
    #     knowledge_base = self._build_knowledge_base(
    #         buggy_methods,
    #         failing_tests,
    #         bug_info
    #     )
        
    #     # Initialize empty dynamic context
    #     dynamic_context = {}
        
    #     # Main repair loop for multi-method fixing
    #     for iteration in range(5):
    #         self.logger.info(f"{'='*60}")
    #         self.logger.info(f"Multi-Method Iteration {iteration + 1}/{self.config.max_iterations}")
    #         self.metrics['total_iterations'] += 1
            
    #         if monitor:
    #             monitor.log_iteration_start(iteration + 1)
            
    #         # Update dynamic context after first failure
    #         if iteration > 0:
    #             self.logger.info("Requesting LLM decision for multi-method context update...")
                
    #             # Combine all buggy method codes for context decision
    #             combined_buggy_code = "\n\n".join([
    #                 f"// Method: {m.method_name}\n{m.buggy_code if hasattr(m, 'buggy_code') else ''}"
    #                 for m in buggy_methods
    #             ])
                
    #             # Get LLM decision
    #             decision = self.context_manager.decide_context_update(
    #                 buggy_method=combined_buggy_code,
    #                 tried_hypotheses=self.hypothesis_pool.get_recent_hypotheses(),
    #                 current_context=dynamic_context,
    #                 retrieval_history=self.context_manager.state.retrieval_history
    #             )
                
    #             # Execute decisions for all methods
    #             dynamic_context = self.context_manager.execute_context_decisions(
    #                 current_context=dynamic_context,
    #                 decision=decision,
    #                 tools=self.tools,
    #                 bug_info={
    #                     'bug_id': bug_info.get('bug_id'),
    #                     'buggy_methods': [m.buggy_code for m in buggy_methods if hasattr(m, 'buggy_code')],
    #                     'buggy_method_names': [m.method_name for m in buggy_methods],
    #                     'buggy_file_paths': list(set([m.file_path for m in buggy_methods])),
    #                     'failing_tests': failing_tests,
    #                     'project_path': self.executor.project_path
    #                 }
    #             )
                
    #             if monitor:
    #                 monitor.log_context_update(
    #                     dynamic_context,
    #                     dynamic_context,
    #                     ', '.join([cmd['tool'] for cmd in decision.get('tool_commands', [])])
    #                 )
            
    #         # Generate prompt for multi-method fixing
    #         prompt = self._create_multi_method_prompt(
    #             buggy_methods=buggy_methods,
    #             knowledge_base=knowledge_base,
    #             dynamic_context=dynamic_context,
    #             iteration=iteration + 1,
    #             recent_hypotheses=self.hypothesis_pool.get_recent_hypotheses() if iteration > 0 else []
    #         )
            
    #         # Generate patches for ALL methods
    #         patch_hypothesises = self.generator.generate_multi_patch(prompt, buggy_methods, 10)
            
    #         for idx, patch_hypothesis in enumerate(patch_hypothesises, 1):
    #             # Parse multi-method patch response
    #             multi_patches = self._parse_multi_method_patch(patch_hypothesis, buggy_methods)
                
    #             # Store original codes for overfitting detection
    #             multi_patches['original_codes'] = {
    #                 m.method_name: m.buggy_code if hasattr(m, 'buggy_code') else ''
    #                 for m in buggy_methods
    #             }
                
    #             if monitor:
    #                 monitor.log_hypothesis(multi_patches)
                
    #             # Execute all patches together
    #             execution_result = self.executor.execute_multi_patches(
    #                 multi_patches,
    #                 failing_tests
    #             )
                
    #             if monitor:
    #                 monitor.log_execution_result(execution_result)
            
    #             # Check result
    #             if execution_result['status'] == 'success':
    #                 self.logger.info(f"✓ All {len(buggy_methods)} methods pass tests!")
                    
    #                 # NEW: Check for overfitting in multi-method patches
    #                 if self.config.enable_overfitting_detection:
    #                     is_overfitting, overfitting_analysis = self._detect_multi_method_overfitting(
    #                         multi_patches, bug_info, failing_tests
    #                     )
                        
    #                     if is_overfitting:
    #                         # Try refinement
    #                         refined_patches = self._refine_multi_method_overfitting(
    #                             multi_patches, overfitting_analysis, bug_info, 
    #                             failing_tests, buggy_methods
    #                         )
                            
    #                         if refined_patches:
    #                             self.logger.info(f"✅ Successfully refined multi-method patches!")
    #                             self.metrics['successful_patches'] += 1
    #                             return refined_patches
    #                         else:
    #                             self.logger.warning("Failed to refine multi-method overfitting")
    #                             continue
    #                     else:
    #                         self.logger.info(f"✅ SUCCESS: All methods semantically correct!")
    #                         self.metrics['successful_patches'] += 1
    #                         return multi_patches
    #                 else:
    #                     self.logger.info(f"✅ SUCCESS fixing all {len(buggy_methods)} methods!")
    #                     self.metrics['successful_patches'] += 1
    #                     return multi_patches
    #             else:
    #                 self.logger.warning(f"✗ Failed: {execution_result.get('error_type')}")
    #                 self.metrics['failed_patches'] += 1
                    
    #                 # Update hypothesis pool
    #                 insights = self.hypothesis_updater.analyze_and_update(
    #                     multi_patches,
    #                     execution_result,
    #                     self.hypothesis_pool
    #                 )
            
    #         if self._should_stop_early(iteration, insights):
    #             self.logger.info("Stopping early based on insights")
    #             break
        
    #     self.logger.warning(f"Failed to fix {len(buggy_methods)} methods after iterations")
    #     return None
    
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
            refined_patches = self.generator.generate_multi_patch(prompt, buggy_methods, 3)
            
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
        
        # Validate patches
        success = len(patches) > 0 and all(p is not None for p in patches)
        
        # Generate report
        monitor.generate_report()
        
        return {
            'success': success,
            'patches': patches,
            'bug_id': bug_id,
            'iterations': self.metrics['total_iterations'],
            'monitor_file': f"traces/{bug_id}_trace.json"
        }
    
    def _process_components(self, components: List[List[Tuple]], bug_info: Dict) -> List[Dict]:
        """Process all components (can be single or multiple)."""
        patches = []
        
        if self.config.parallel_components and len(components) > 1:
            # Process in parallel
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(components), 4)) as executor:
                futures = []
                for component in components:
                    future = executor.submit(
                        self._process_component,
                        component,
                        bug_info
                    )
                    futures.append(future)
                
                for future in futures:
                    result = future.result()
                    if result:
                        patches.append(result)
        else:
            # Sequential processing
            for i, component in enumerate(components):
                self.logger.info(f"Processing component {i+1}/{len(components)}")
                patch = self._process_component(component, bug_info)
                if patch:
                    patches.append(patch)
        
        return patches
    
    def _process_component(self, component: List[Tuple], bug_info: Dict) -> Optional[Dict]:
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
        
        # Determine if this is truly multi-method or single-method
        if len(buggy_methods) == 1:
            return self._process_single_method(buggy_methods[0], failing_tests, bug_info)
        else:
            return self._process_multi_method(buggy_methods, failing_tests, bug_info)
    
    # def _process_single_method(self, buggy_method: FaultLocation, 
    #                            failing_tests: List[str], bug_info: Dict) -> Optional[Dict]:
    #     """Process a single buggy method with its failing tests."""
    #     self.logger.info(f"Processing single method: {buggy_method.method_name}")
        
    #     # Get monitor
    #     monitor = self.monitors.get(bug_info.get('bug_id'))
        
    #     # Build knowledge base for single method
    #     knowledge_base = self._build_knowledge_base(
    #         [buggy_method], 
    #         failing_tests,
    #         bug_info
    #     )
        
    #     # Initialize empty dynamic context
    #     dynamic_context = {}
        
    #     # Main repair loop
    #     for iteration in range(3):
    #         self.logger.info(f"{'='*60}")
    #         self.logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")
    #         self.metrics['total_iterations'] += 1
            
    #         if monitor:
    #             monitor.log_iteration_start(iteration + 1)
            
    #         # Update dynamic context after first failure
    #         if iteration > 0:
    #             self.logger.info("Requesting LLM decision for context update...")
                
    #             # Get LLM decision on what to retrieve
    #             decision = self.context_manager.decide_context_update(
    #                 buggy_method=buggy_method.buggy_code if hasattr(buggy_method, 'buggy_code') else '',
    #                 tried_hypotheses=self.hypothesis_pool.get_recent_hypotheses(),
    #                 current_context=dynamic_context,
    #                 retrieval_history=self.context_manager.state.retrieval_history
    #             )
    #             if monitor:
    #                 monitor.log_llm_response(
    #                     response=json.dumps(decision),
    #                     response_type="context_decision",
    #                     metadata={
    #                         'tools_selected': [cmd['tool'] for cmd in decision.get('tool_commands', [])],
    #                         'reasoning_length': len(decision.get('reasoning', '')),
    #                         'focus_area': decision.get('focus', '')
    #                     }
    #                 )
    #             self.logger.info(f"LLM decided to use tools: {[cmd['tool'] for cmd in decision.get('tool_commands', [])]}")
                
    #             # Execute LLM decisions with validated commands
    #             dynamic_context = self.context_manager.execute_context_decisions(
    #                 current_context=dynamic_context,
    #                 decision=decision,
    #                 tools=self.tools,
    #                 bug_info={
    #                     'bug_id': bug_info.get('bug_id'),
    #                     'buggy_method': buggy_method.buggy_code,
    #                     'buggy_method_name': buggy_method.method_name,
    #                     'buggy_class': buggy_method.class_name,
    #                     'buggy_file_path': buggy_method.file_path,
    #                     'failing_tests': failing_tests,
    #                     'project_path': self.executor.project_path
    #                 }
    #             )
                
    #             if monitor:
    #                 monitor.log_context_update(
    #                     dynamic_context,
    #                     dynamic_context,
    #                     ', '.join([cmd['tool'] for cmd in decision.get('tool_commands', [])])
    #                 )
                
    #             self.logger.info(f"Dynamic context updated: {len(dynamic_context)} items")
            
    #         # Generate prompt for single method
    #         prompt = self._create_single_method_prompt(
    #             buggy_method=buggy_method,
    #             knowledge_base=knowledge_base,
    #             dynamic_context=dynamic_context,
    #             iteration=iteration + 1,
    #             recent_hypotheses=self.hypothesis_pool.get_recent_hypotheses() if iteration > 0 else []
    #         )
    #         if monitor:
    #             monitor.log_prompt(
    #                 prompt=prompt,
    #                 prompt_type="generation",  # or "context_decision", "analysis"
    #                 metadata={
    #                     'iteration': iteration + 1,
    #                     'method_name': buggy_method.method_name,
    #                     'failing_tests_count': len(failing_tests),
    #                     'context_size': len(dynamic_context),
    #                     'hypothesis_count': len(self.hypothesis_pool.get_recent_hypotheses() if iteration > 0 else [])
    #                 }
    #             )
        
    #         # Generate patch
    #         patch_hypothesises = self.generator.generate_patch(prompt, buggy_method.buggy_code if hasattr(buggy_method, 'buggy_code') else '', 10)
            
    #         # Add metadata
    #         for patch_hypothesis in patch_hypothesises:
    #             patch_hypothesis['file_path'] = buggy_method.file_path
    #             patch_hypothesis['method_name'] = buggy_method.method_name
    #             patch_hypothesis['class_name'] = buggy_method.class_name
    #             patch_hypothesis['iteration'] = iteration + 1
    #             patch_hypothesis['is_multi_method'] = False
    #             patch_hypothesis['line_numbers'] = knowledge_base.get('buggy_methods', [{}])[0].get('lines', [])

            
    #             if monitor:
    #                 monitor.log_hypothesis(patch_hypothesis)
            
    #         # Execute patch
    #         execution_results = []

    #         for idx, patch_hypothesis in enumerate(patch_hypothesises, 1):
    #             print(patch_hypothesis.get('fixed_method', ''))
    #             execution_result = self.executor.execute_patch(
    #                 patch_hypothesis,
    #                 failing_tests
    #             )
                
    #             if monitor:
    #                 monitor.log_execution_result(execution_result)

    #             # Save execution result for later analysis
    #             execution_results.append({
    #                 "patch_id": idx,
    #                 "patch": patch_hypothesis,
    #                 "result": execution_result
    #             })

    #             # Check result
    #             if execution_result["status"] == "success":
    #                 self.logger.info(f"✓ SUCCESS with patch {idx} at iteration {iteration + 1}!")
    #                 self.metrics["successful_patches"] += 1
    #                 return patch_hypothesis
    #             else:
    #                 self.logger.warning(f"✗ Failed (patch {idx}): {execution_result.get('error_type')}")
    #                 self.metrics["failed_patches"] += 1

    #                 # Update hypothesis pool
    #                 insights = self.hypothesis_updater.analyze_and_update(
    #                     patch_hypothesis,
    #                     execution_result,
    #                     self.hypothesis_pool
    #                 )

    #         if self._should_stop_early(iteration, insights):
    #             self.logger.info("Stopping early based on insights")
    #             break

    #     self.logger.warning(
    #         f"Failed to fix single method after {self.config.max_iterations} iterations "
    #         f"and {len(patch_hypothesises)} candidates"
    #     )
    #     return None
    
    # def _process_multi_method(self, buggy_methods: List[FaultLocation], 
    #                          failing_tests: List[str], bug_info: Dict) -> Optional[Dict]:
    #     """
    #     Process multiple buggy methods that need to be fixed together.
    #     This is the critical case where methods are interdependent.
    #     """
    #     self.logger.info(f"Processing {len(buggy_methods)} interdependent buggy methods")
        
    #     # Get monitor
    #     monitor = self.monitors.get(bug_info.get('bug_id'))
        
    #     # Build knowledge base for ALL methods
    #     knowledge_base = self._build_knowledge_base(
    #         buggy_methods,  # ALL methods
    #         failing_tests,
    #         bug_info
    #     )
        
    #     # Initialize empty dynamic context
    #     dynamic_context = {}
        
    #     # Main repair loop for multi-method fixing
    #     for iteration in range(3):
    #         self.logger.info(f"{'='*60}")
    #         self.logger.info(f"Multi-Method Iteration {iteration + 1}/{self.config.max_iterations}")
    #         self.metrics['total_iterations'] += 1
            
    #         if monitor:
    #             monitor.log_iteration_start(iteration + 1)
            
    #         # Update dynamic context after first failure
    #         if iteration > 0:
    #             self.logger.info("Requesting LLM decision for multi-method context update...")
                
    #             # Combine all buggy method codes for context decision
    #             combined_buggy_code = "\n\n".join([
    #                 f"// Method: {m.method_name}\n{m.buggy_code if hasattr(m, 'buggy_code') else ''}"
    #                 for m in buggy_methods
    #             ])
                
    #             # Get LLM decision
    #             decision = self.context_manager.decide_context_update(
    #                 buggy_method=combined_buggy_code,
    #                 tried_hypotheses=self.hypothesis_pool.get_recent_hypotheses(),
    #                 current_context=dynamic_context,
    #                 retrieval_history=self.context_manager.state.retrieval_history
    #             )
                
    #             # Execute decisions for all methods
    #             dynamic_context = self.context_manager.execute_context_decisions(
    #                 current_context=dynamic_context,
    #                 decision=decision,
    #                 tools=self.tools,
    #                 bug_info={
    #                     'bug_id':bug_info.get('bug_id'),
    #                     'buggy_methods': [m.buggy_code for m in buggy_methods if hasattr(m, 'buggy_code')],
    #                     'buggy_method_names': [m.method_name for m in buggy_methods],
    #                     'buggy_file_paths': list(set([m.file_path for m in buggy_methods])),
    #                     'failing_tests': failing_tests,
    #                     'project_path': self.executor.project_path
    #                 }
    #             )
                
    #             if monitor:
    #                 monitor.log_context_update(
    #                     dynamic_context,
    #                     dynamic_context,
    #                     ', '.join([cmd['tool'] for cmd in decision.get('tool_commands', [])])
    #                 )
            
    #         # Generate prompt for multi-method fixing
    #         prompt = self._create_multi_method_prompt(
    #             buggy_methods=buggy_methods,
    #             knowledge_base=knowledge_base,
    #             dynamic_context=dynamic_context,
    #             iteration=iteration + 1,
    #             recent_hypotheses=self.hypothesis_pool.get_recent_hypotheses() if iteration > 0 else []
    #         )
            
    #         # Generate patches for ALL methods
    #         patch_hypothesises = self.generator.generate_multi_patch(prompt, buggy_methods, 10)
            
    #         for idx, patch_hypothesis in enumerate(patch_hypothesises, 1):
    #         # Parse multi-method patch response
    #             multi_patches = self._parse_multi_method_patch(patch_hypothesis, buggy_methods)
                
    #             if monitor:
    #                 monitor.log_hypothesis(multi_patches)
                
    #             # Execute all patches together
    #             execution_result = self.executor.execute_multi_patches(
    #                 multi_patches,
    #                 failing_tests
    #             )
                
    #             if monitor:
    #                 monitor.log_execution_result(execution_result)
            
    #             # Check result
    #             if execution_result['status'] == 'success':
    #                 self.logger.info(f"✓ SUCCESS fixing all {len(buggy_methods)} methods at iteration {iteration + 1}!")
    #                 self.metrics['successful_patches'] += 1
    #                 return multi_patches
    #             else:
    #                 self.logger.warning(f"✗ Failed: {execution_result.get('error_type')}")
    #                 self.metrics['failed_patches'] += 1
                    
    #                 # Update hypothesis pool
    #                 insights = self.hypothesis_updater.analyze_and_update(
    #                     multi_patches,
    #                     execution_result,
    #                     self.hypothesis_pool
    #                 )
                    
    #         if self._should_stop_early(iteration, insights):
    #             self.logger.info("Stopping early based on insights")
    #             break
        
    #     self.logger.warning(f"Failed to fix {len(buggy_methods)} methods after {self.config.max_iterations} iterations")
    #     return None
    
    def _create_single_method_prompt(self, buggy_method: FaultLocation, knowledge_base: Dict,
                                    dynamic_context: Dict, iteration: int, 
                                    recent_hypotheses: List[Dict]) -> str:
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
                prompt += f"""
Attempt {i}:
 #### Applied Changes: 
    {hyp.get('changes', 'N/A')}
 #### Result: {hyp.get('failure_reason', {})}
"""     
        # Add dynamic context
        if dynamic_context:
            prompt += "\n### RETRIEVED CONTEXT:\n"
            for key, value in list(dynamic_context.items())[:5]:
                prompt += f"\n#### {key.replace('_', ' ').title()}:\n"
                if isinstance(value, dict) and 'data' in value:
                    
                    prompt += f"{json.dumps(value['data'], indent=2)}\n"[:2000]
        
        prompt += """
### TASK:
Fix the buggy method to make all failing tests pass.

Provide your response in JSON format:
{
    "hypothesis": "Short explanation of the bug and fix",
    "fixed_method": "Complete fixed method code"
}
"""     
        return prompt
    
    def _create_multi_method_prompt(self, buggy_methods: List[FaultLocation], knowledge_base: Dict,
                                   dynamic_context: Dict, iteration: int, 
                                   recent_hypotheses: List[Dict]) -> str:
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
                prompt += f"""
Attempt {i}:
- Hypothesis: {hyp.get('hypothesis', 'N/A')}
- Changes: {hyp.get('changes', 'N/A')}
- Result: 
    - Still failing tests : {hyp.get('execution_result', '').get('failed_tests', '')}
    - Passed tests : {hyp.get('execution_result', '').get('passed_tests', '')}
"""
        
        # Add dynamic context
        if dynamic_context:
            prompt += "\n### RETRIEVED CONTEXT:\n"
            for key, value in list(dynamic_context.items())[:5]:
                prompt += f"\n#### {key.replace('_', ' ').title()}:\n"
                if isinstance(value, dict) and 'data' in value:
                    prompt += f"{str(json.dumps(value['data'], indent=2))[:3000]}\n"
        prompt += f"""

### TASK:
Provide fixes for ALL {len(buggy_methods)} methods that work together.

Return JSON with fixes for ALL methods:
{{
    "hypothesis": "Overall explanation of the bug and coordinated fix strategy",
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
    "changes": "applied changes in specific format"
}}
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
                        func_info.get('end_loc', 0) + 1
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
        method_pattern = re.compile(
            r"""
            (?:@\w+(?:\([^)]*\))?\s+)*       # Annotations (e.g., @Override, @SuppressWarnings("x"))
            (?:public|private|protected)?\s* # Access modifier
            (?:static\s+)?                   # Optional static
            (?:final\s+)?                    # Optional final
            (?:synchronized\s+)?             # Optional synchronized
            (?:[\w<>\[\],?]+\s+)+            # Return type (with generics/arrays)
            (?P<name>\w+)                    # Method name
            \s*\(                            # Opening parenthesis
            (?P<params>[^)]*)                # Parameter list
            \)                               # Closing parenthesis
            """,
            re.VERBOSE | re.MULTILINE | re.DOTALL
        )

        match = method_pattern.search(java_code)
        if not match:
            return None

        method_name = match.group("name")
        params = match.group("params").strip()

        if not params:
            return {"name": method_name, "num_params": 0, "param_types": []}

        # Split parameters (handle generics with <>, arrays, etc.)
        raw_params = [p.strip() for p in re.split(r",(?![^<]*>)", params) if p.strip()]

        param_types = []
        for p in raw_params:
            # Remove vararg (...) and parameter name
            parts = p.split()
            if not parts:
                continue
            # All but last token is the type, last token is usually the variable name
            if len(parts) > 1:
                ptype = " ".join(parts[:-1])
            else:
                ptype = parts[0]  # in case no variable name (rare but possible)
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
        # # For multiple fault locations, check if they should be grouped
        # # This is a simplified version - should use coverage analysis
        
        # # Check if all fault locations are in the same file or class
        # files = set(fl.file_path for fl in fault_locations)
        # classes = set(fl.class_name for fl in fault_locations)
        
        # if len(files) == 1 or len(classes) == 1:
        #     # All in same file/class - likely interdependent
        #     # Create single component with all methods and all tests
        #     component = []
        #     for fl in fault_locations:
        #         for test in failing_tests:
        #             component.append((fl, test))
        #     return [component]
        # else:
        #     # Different files/classes - might be independent
        #     # For now, treat as single component (conservative approach)
        #     # Better approach would use coverage to determine actual dependencies
        #     component = []
        #     for fl in fault_locations:
        #         for test in failing_tests:
        #             component.append((fl, test))
        #     return [component]
    
    def _should_stop_early(self, iteration: int, insights: Dict) -> bool:
        """Determine if should stop early based on insights."""
        # Stop if no progress after many iterations
        if iteration > 10 and insights.get('no_progress'):
            return True
     
    def _run_single_test_for_error(self, test_name: str, language: str) -> Dict:
        """Run a single test to get error information."""
        # This would run the actual test
        # Placeholder implementation
        return {
            'error_type': 'AssertionError',
            'error_message': 'Test failed',
            'stack_trace': 'Stack trace here'
        }
    
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
    Enhanced monitor that tracks and saves complete bug fixing progress.
    Now includes prompts, tool executions, and context snapshots.
    """
    
    def __init__(self, bug_id: str, output_dir: str = "traces"):
        self.bug_id = bug_id
        self.output_dir = output_dir
        self.start_time = time.time()
        
        # Core tracking data
        self.iteration_data = []
        self.context_evolution = []
        self.hypothesis_evolution = []
        self.tool_usage = []
        self.error_patterns = {}
        
        # New tracking for complete data
        self.prompts_log = []
        self.tool_executions_log = []
        self.context_snapshots = []
        self.llm_responses = []
        
        # Current iteration placeholder
        self.current_iteration = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup detailed logging
        self.logger = logging.getLogger(f"Monitor_{bug_id}")
        handler = logging.FileHandler(f"{output_dir}/{bug_id}_detailed.log")
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.info(f"Monitor initialized for bug: {bug_id}")
    
    def log_iteration_start(self, iteration: int):
        """Log the start of an iteration with complete structure."""
        self.current_iteration = {
            'iteration': iteration,
            'start_time': time.time(),
            'timestamp': datetime.now().isoformat(),
            'prompts': [],
            'tool_executions': [],
            'context_snapshot': {},
            'llm_responses': [],
            'metrics': {
                'prompt_tokens': 0,
                'response_tokens': 0,
                'tool_calls': 0,
                'context_size': 0
            }
        }
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting Iteration {iteration}")
        self.logger.info(f"{'='*60}")
    
    def log_prompt(self, prompt: str, prompt_type: str = "generation", metadata: Dict = None):
        """
        Log prompts sent to LLM with metadata.
        
        Args:
            prompt: The actual prompt text
            prompt_type: Type of prompt (generation, context_update, analysis)
            metadata: Additional metadata about the prompt
        """
        prompt_entry = {
            'type': prompt_type,
            'content': prompt,
            'timestamp': time.time(),
            'token_estimate': self._estimate_tokens(prompt),
            'metadata': metadata or {}
        }
        
        if 'prompts' not in self.current_iteration:
            self.current_iteration['prompts'] = []
        
        self.current_iteration['prompts'].append(prompt_entry)
        self.prompts_log.append({
            'iteration': self.current_iteration.get('iteration', 0),
            **prompt_entry
        })
        
        # Update metrics
        self.current_iteration['metrics']['prompt_tokens'] += prompt_entry['token_estimate']
        
        self.logger.debug(f"Logged {prompt_type} prompt: {len(prompt)} chars, ~{prompt_entry['token_estimate']} tokens")
    
    def log_llm_response(self, response: str, response_type: str = "patch", metadata: Dict = None):
        """
        Log LLM responses with metadata.
        
        Args:
            response: The LLM response text
            response_type: Type of response (patch, decision, analysis)
            metadata: Additional metadata about the response
        """
        response_entry = {
            'type': response_type,
            'content': response[:5000] if len(response) > 5000 else response,  # Truncate very long responses
            'timestamp': time.time(),
            'token_estimate': self._estimate_tokens(response),
            'metadata': metadata or {},
            'truncated': len(response) > 5000
        }
        
        if 'llm_responses' not in self.current_iteration:
            self.current_iteration['llm_responses'] = []
        
        self.current_iteration['llm_responses'].append(response_entry)
        self.llm_responses.append({
            'iteration': self.current_iteration.get('iteration', 0),
            **response_entry
        })
        
        # Update metrics
        self.current_iteration['metrics']['response_tokens'] += response_entry['token_estimate']
        
        self.logger.debug(f"Logged {response_type} response: {len(response)} chars, ~{response_entry['token_estimate']} tokens")
    
    def log_tool_execution(self, tool_name: str, params: Dict, result: Any, execution_time: float = None):
        """
        Log detailed tool execution with parameters and results.
        
        Args:
            tool_name: Name of the tool executed
            params: Parameters passed to the tool
            result: Result returned by the tool
            execution_time: Time taken to execute (optional)
        """
        # Serialize result safely
        try:
            if isinstance(result, (dict, list, str, int, float, bool, type(None))):
                serialized_result = result
            else:
                serialized_result = str(result)
        except:
            serialized_result = "Result could not be serialized"
        
        # Truncate large results
        if isinstance(serialized_result, str) and len(serialized_result) > 2000:
            serialized_result = serialized_result[:2000] + "... [truncated]"
        elif isinstance(serialized_result, dict):
            # Truncate large dict values
            serialized_result = self._truncate_dict(serialized_result, 2000)
        
        tool_execution = {
            'tool': tool_name,
            'params': params,
            'result': serialized_result,
            'timestamp': time.time(),
            'execution_time': execution_time,
            'success': result is not None
        }
        
        if 'tool_executions' not in self.current_iteration:
            self.current_iteration['tool_executions'] = []
        
        self.current_iteration['tool_executions'].append(tool_execution)
        self.tool_executions_log.append({
            'iteration': self.current_iteration.get('iteration', 0),
            **tool_execution
        })
        
        # Also add to general tool usage for backward compatibility
        self.tool_usage.append({
            'iteration': self.current_iteration.get('iteration', 0),
            'tool': tool_name,
            'timestamp': time.time(),
            'params_summary': self._summarize_params(params)
        })
        
        # Update metrics
        self.current_iteration['metrics']['tool_calls'] += 1
        
        self.logger.debug(f"Tool executed: {tool_name} with params: {self._summarize_params(params)}")
        if execution_time:
            self.logger.debug(f"Execution time: {execution_time:.2f}s")
    
    def log_context_update(self, context: Dict, new_info: Dict, tool_used: str = None):
        """
        Enhanced context update logging with full snapshot capability.
        
        Args:
            context: Current full context
            new_info: New information added
            tool_used: Tool that was used to get the info
        """
        context_size = self._estimate_tokens(context)
        
        # Create context snapshot (truncated for storage)
        snapshot = self._create_context_snapshot(context)
        
        context_entry = {
            'iteration': self.current_iteration.get('iteration', 0),
            'timestamp': time.time(),
            'context_size_tokens': context_size,
            'new_info_keys': list(new_info.keys()),
            'new_info_summary': self._summarize_dict(new_info, max_depth=2),
            'tool_used': tool_used,
            'snapshot': snapshot
        }
        
        self.context_evolution.append(context_entry)
        
        # Store snapshot
        self.current_iteration['context_snapshot'] = snapshot
        self.context_snapshots.append({
            'iteration': self.current_iteration.get('iteration', 0),
            'snapshot': snapshot,
            'size_tokens': context_size
        })
        
        # Update metrics
        self.current_iteration['metrics']['context_size'] = context_size
        
        self.logger.debug(f"Context updated - Size: {context_size} tokens, Tool: {tool_used}")
        self.logger.debug(f"New keys added: {list(new_info.keys())}")
    
    def log_hypothesis(self, hypothesis: Dict):
        """
        Log hypothesis generation with complete details.
        
        Args:
            hypothesis: Hypothesis dictionary with all details
        """
        hypothesis_entry = {
            'iteration': self.current_iteration.get('iteration', 0),
            'hypothesis': hypothesis.get('hypothesis', ''),
            'changes': hypothesis.get('changes', ''),
            'approach_type': hypothesis.get('approach_type', 'unknown'),
            'fixed_method': hypothesis.get('fixed_method', ''), 
            'is_multi_method': hypothesis.get('is_multi_method', False),
            'methods_count': hypothesis.get('methods_count', 1),
            'confidence': hypothesis.get('confidence', 0),
            'timestamp': time.time()
        }
        
        self.hypothesis_evolution.append(hypothesis_entry)
        
        # Store in current iteration
        self.current_iteration['hypothesis'] = hypothesis_entry
        
        self.logger.info(f"Generated hypothesis: {hypothesis.get('hypothesis', '')[:200]}")
        self.logger.debug(f"Approach: {hypothesis_entry['approach_type']}")
    
    def log_execution_result(self, result: Dict):
        """
        Log execution result with complete details.
        
        Args:
            result: Execution result dictionary
        """
        self.current_iteration['execution_result'] = result
        self.current_iteration['end_time'] = time.time()
        self.current_iteration['duration'] = self.current_iteration['end_time'] - self.current_iteration['start_time']
        
        # Add complete iteration data
        self.iteration_data.append(self.current_iteration)
        
        # Track error patterns
        error_type = result.get('error_type', 'Unknown')
        self.error_patterns[error_type] = self.error_patterns.get(error_type, 0) + 1
        
        # Log result
        status = result.get('status', 'unknown')
        self.logger.info(f"Execution result: {status}")
        if status == 'failed':
            self.logger.info(f"Error type: {error_type}")
            self.logger.debug(f"Error message: {result.get('error_message', 'N/A')[:500]}")
        
        # Save intermediate results periodically
        if len(self.iteration_data) % 5 == 0:
            self._save_intermediate_report()
    
    def _create_context_snapshot(self, context: Dict, max_size: int = 5000) -> Dict:
        """
        Create a snapshot of the context for storage.
        
        Args:
            context: Full context dictionary
            max_size: Maximum size for snapshot
            
        Returns:
            Truncated snapshot suitable for storage
        """
        snapshot = {}
        current_size = 0
        
        for key, value in context.items():
            # Estimate size of this entry
            entry_size = len(json.dumps({key: value}, default=str))
            
            if current_size + entry_size > max_size:
                # Add a marker that there's more
                snapshot['_truncated'] = True
                snapshot['_truncated_keys'] = list(context.keys())[len(snapshot):]
                break
            
            # Add to snapshot
            if isinstance(value, (dict, list)):
                snapshot[key] = self._truncate_value(value, 500)
            else:
                snapshot[key] = value
            
            current_size += entry_size
        
        return snapshot
    
    def _truncate_value(self, value: Any, max_size: int) -> Any:
        """Truncate a value for storage."""
        if isinstance(value, str):
            if len(value) > max_size:
                return value[:max_size] + "... [truncated]"
            return value
        elif isinstance(value, dict):
            return self._truncate_dict(value, max_size)
        elif isinstance(value, list):
            if len(value) > 10:
                return value[:10] + ["... truncated"]
            return value
        else:
            return value
    
    def _truncate_dict(self, d: Dict, max_size: int) -> Dict:
        """Truncate dictionary values for storage."""
        result = {}
        for key, value in d.items():
            if isinstance(value, str) and len(value) > max_size:
                result[key] = value[:max_size] + "... [truncated]"
            elif isinstance(value, (dict, list)):
                result[key] = str(value)[:max_size] if len(str(value)) > max_size else value
            else:
                result[key] = value
        return result
    
    def _summarize_dict(self, d: Dict, max_depth: int = 1, current_depth: int = 0) -> Dict:
        """Create a summary of a dictionary."""
        if current_depth >= max_depth:
            return {k: type(v).__name__ for k, v in d.items()}
        
        summary = {}
        for key, value in d.items():
            if isinstance(value, dict):
                summary[key] = self._summarize_dict(value, max_depth, current_depth + 1)
            elif isinstance(value, list):
                summary[key] = f"List[{len(value)} items]"
            elif isinstance(value, str):
                summary[key] = value[:100] + "..." if len(value) > 100 else value
            else:
                summary[key] = value
        
        return summary
    
    def _summarize_params(self, params: Dict) -> str:
        """Create a brief summary of parameters."""
        if not params:
            return "{}"
        
        items = []
        for key, value in params.items():
            if isinstance(value, str):
                val_str = f'"{value[:20]}..."' if len(value) > 20 else f'"{value}"'
            elif isinstance(value, list):
                val_str = f"[{len(value)} items]"
            elif isinstance(value, dict):
                val_str = f"{{...}}"
            else:
                val_str = str(value)
            items.append(f"{key}={val_str}")
        
        return "{" + ", ".join(items[:3]) + ("..." if len(items) > 3 else "") + "}"
    
    def _save_intermediate_report(self):
        """Save intermediate report to avoid data loss."""
        intermediate_file = f"{self.output_dir}/{self.bug_id}_intermediate.json"
        
        try:
            report = self._create_report()
            with open(intermediate_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.debug(f"Saved intermediate report to {intermediate_file}")
        except Exception as e:
            self.logger.error(f"Failed to save intermediate report: {e}")
    
    def _create_report(self) -> Dict:
        """Create comprehensive report structure."""
        return {
            'bug_id': self.bug_id,
            'total_time': time.time() - self.start_time,
            'iterations': len(self.iteration_data),
            'successful': any(it.get('execution_result', {}).get('status') == 'success' 
                            for it in self.iteration_data),
            
            # Summary statistics
            'error_patterns': self.error_patterns,
            'tool_usage_summary': self._summarize_tool_usage(),
            'context_growth': self._analyze_context_growth(),
            'hypothesis_summary': self._summarize_hypotheses(),
            
            # Complete detailed data
            'iteration_details': self.iteration_data,
            'prompts_log': self.prompts_log,
            'tool_executions_log': self.tool_executions_log,
            'context_snapshots': self.context_snapshots,
            'hypothesis_evolution': self.hypothesis_evolution,
            
            # Metrics
            'total_metrics': self._calculate_total_metrics()
        }
    
    def generate_report(self):
        """
        Generate and save comprehensive final report.
        
        Returns:
            Dict: Complete report data
        """
        report = self._create_report()
        
        # Save complete report
        report_file = f"{self.output_dir}/{self.bug_id}_trace.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Complete report saved to {report_file}")
        
        # Also save a human-readable summary
        self._save_summary(report)
        
        # Clean up intermediate file if it exists
        intermediate_file = f"{self.output_dir}/{self.bug_id}_intermediate.json"
        if os.path.exists(intermediate_file):
            os.remove(intermediate_file)
        
        return report
    
    def _save_summary(self, report: Dict):
        """Save human-readable summary."""
        summary_file = f"{self.output_dir}/{self.bug_id}_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Bug Fixing Summary: {self.bug_id}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Status: {'SUCCESS' if report['successful'] else 'FAILED'}\n")
            f.write(f"Total Time: {report['total_time']:.2f} seconds\n")
            f.write(f"Iterations: {report['iterations']}\n\n")
            
            f.write("Error Distribution:\n")
            for error, count in report['error_patterns'].items():
                f.write(f"  - {error}: {count}\n")
            
            f.write("\nTool Usage:\n")
            for tool, count in report['tool_usage_summary'].items():
                f.write(f"  - {tool}: {count} times\n")
            
            f.write("\nMetrics Summary:\n")
            metrics = report['total_metrics']
            f.write(f"  - Total prompts: {metrics['total_prompts']}\n")
            f.write(f"  - Total tokens: ~{metrics['total_tokens']}\n")
            f.write(f"  - Tool executions: {metrics['total_tool_calls']}\n")
            f.write(f"  - Unique approaches: {metrics['unique_approaches']}\n")
        
        self.logger.info(f"Summary saved to {summary_file}")
    
    def _estimate_tokens(self, data: Any) -> int:
        """Estimate token count for any data."""
        if data is None:
            return 0
        text = json.dumps(data, default=str) if not isinstance(data, str) else data
        # Rough estimation: 1 token per 4 characters
        return len(text) // 4
    
    def _summarize_tool_usage(self) -> Dict:
        """Summarize tool usage statistics."""
        tool_counts = {}
        for usage in self.tool_usage:
            tool = usage['tool']
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        return tool_counts
    
    def _analyze_context_growth(self) -> Dict:
        """Analyze how context grew over iterations."""
        if not self.context_evolution:
            return {'growth_rate': 0, 'final_size': 0}
        
        sizes = [c['context_size_tokens'] for c in self.context_evolution]
        
        return {
            'initial_size': sizes[0] if sizes else 0,
            'final_size': sizes[-1] if sizes else 0,
            'max_size': max(sizes) if sizes else 0,
            'growth_rate': (sizes[-1] - sizes[0]) / len(sizes) if len(sizes) > 1 else 0,
            'iterations_with_growth': sum(1 for i in range(1, len(sizes)) if sizes[i] > sizes[i-1])
        }
    
    def _summarize_hypotheses(self) -> Dict:
        """Summarize hypothesis patterns."""
        if not self.hypothesis_evolution:
            return {}
        
        approaches = {}
        multi_method_count = 0
        
        for hyp in self.hypothesis_evolution:
            approach = hyp.get('approach_type', 'unknown')
            approaches[approach] = approaches.get(approach, 0) + 1
            
            if hyp.get('is_multi_method'):
                multi_method_count += 1
        
        return {
            'total': len(self.hypothesis_evolution),
            'approach_distribution': approaches,
            'multi_method_fixes': multi_method_count,
            'unique_approaches': len(set(approaches.keys()))
        }
    
    def _calculate_total_metrics(self) -> Dict:
        """Calculate total metrics across all iterations."""
        total_prompts = len(self.prompts_log)
        total_responses = len(self.llm_responses)
        total_tool_calls = len(self.tool_executions_log)
        
        total_prompt_tokens = sum(p.get('token_estimate', 0) for p in self.prompts_log)
        total_response_tokens = sum(r.get('token_estimate', 0) for r in self.llm_responses)
        
        unique_tools = set(t['tool'] for t in self.tool_usage)
        unique_approaches = set(h.get('approach_type') for h in self.hypothesis_evolution)
        
        return {
            'total_prompts': total_prompts,
            'total_responses': total_responses,
            'total_tool_calls': total_tool_calls,
            'total_prompt_tokens': total_prompt_tokens,
            'total_response_tokens': total_response_tokens,
            'total_tokens': total_prompt_tokens + total_response_tokens,
            'unique_tools_used': len(unique_tools),
            'unique_approaches': len(unique_approaches),
            'average_iteration_time': (time.time() - self.start_time) / max(len(self.iteration_data), 1)
        }