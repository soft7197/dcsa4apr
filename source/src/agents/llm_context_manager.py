# src/agents/llm_context_manager.py
"""
Final complete LLM Context Manager with perfect tool integration.
This version ensures seamless operation with all updated tools.
All references to old components removed, full integration with enhanced CodeExtractor.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from openai import OpenAI
import json
import logging
import re
import os
from collections import defaultdict
import hashlib
import time

# Import the smart tool validator
from src.tools.tool_input_validator import ToolInputValidator, ProjectIndexer
from lib.query_vector_db import retrieve_context


@dataclass
class ContextState:
    """Enhanced context state with comprehensive tracking."""
    iteration: int = 0
    retrieved_data: Dict[str, any] = field(default_factory=dict)
    retrieval_history: List[Dict] = field(default_factory=list)
    token_count: int = 0
    best_patch: Optional[Dict] = None
    best_error_count: int = float('inf')
    resolution_history: List[Dict] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    tool_effectiveness: Dict[str, float] = field(default_factory=dict)
    failed_tool_attempts: List[Dict] = field(default_factory=list)
    successful_retrievals: List[Dict] = field(default_factory=list)


class LLMContextManager:
    """
    Complete LLM Context Manager with perfect tool integration.
    Handles all tools properly including the enhanced CodeExtractor.
    No dependencies on old context_updater or context_miner components.
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o", 
                 max_tokens: int = 8000, project_path: str = None):
        """
        Initialize enhanced context manager.
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use
            max_tokens: Maximum token limit for context
            project_path: Project path for smart resolution
        """
        # Initialize OpenAI client
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self.state = ContextState()
        
        # Tool validator and indexer
        self.tool_validator = None
        self.project_indexer = None
        self.project_path = project_path
        
        if project_path:
            self.initialize_smart_resolution(project_path)
        
        # Cache for tool results
        self.tool_cache = {}
        
        # Error pattern to tool mapping
        self.error_to_tool_mapping = self._initialize_error_mapping()
        
        # Token budget allocation
        self.token_budget = {
            'buggy_method': 0.25,
            'error_messages': 0.15,
            'test_code': 0.20,
            'dynamic_context': 0.30,
            'hypotheses': 0.10
        }
        
        self.logger.info(f"LLM Context Manager initialized with model: {model}")
    
    def initialize_smart_resolution(self, project_path: str, bug_info: Dict = None):
        """
        Initialize smart resolution components for tool input validation.
        """
        try:
            self.logger.info(f"Initializing smart resolution for project: {project_path}")
            self.project_indexer = ProjectIndexer(project_path)
            self.tool_validator = ToolInputValidator(
                project_path=project_path,
                bug_info=bug_info or {},
                project_indexer=self.project_indexer
            )
            self.project_path = project_path
            self.logger.info("Smart resolution initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize smart resolution: {e}")
            self.tool_validator = None
            self.project_indexer = None
    
    def update_bug_context(self, bug_info: Dict):
        """Update bug context for better tool parameter resolution."""
        if self.tool_validator:
            self.tool_validator.bug_info = bug_info
        self.current_bug_info = bug_info
    
    def decide_context_update(self, 
                             buggy_method: str,
                             kb,
                             tried_hypotheses: List[Dict],
                             current_context: Dict,
                             retrieval_history: List[Dict] = None) -> Dict:
        """
        Intelligently decide what context to retrieve based on failures.
        
        Args:
            buggy_method: The buggy method code
            tried_hypotheses: List of tried hypotheses with results
            current_context: Current dynamic context
            retrieval_history: History of retrieval attempts
            
        Returns:
            Decision dict with validated tool commands
        """
        self.state.iteration += 1
        self.state.retrieval_history = retrieval_history or []
        
        # Analyze error patterns and failure reasons
        error_patterns = self._analyze_error_patterns(tried_hypotheses)
        failure_analysis = self._analyze_failure_reasons(tried_hypotheses)
        
        # Build intelligent prompt
        prompt = self._build_decision_prompt(
            buggy_method=buggy_method,
            kb = kb,
            tried_hypotheses=tried_hypotheses,
            current_context=current_context,
            retrieval_history=self.state.retrieval_history,
            error_patterns=error_patterns,
            failure_analysis=failure_analysis
        )
        try:
            #Get LLM decision
            response = self.client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {"role": "system", "content": self._get_enhanced_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            #Parse response
            decision = self._parse_llm_response(response.choices[0].message.content)
            self.logger.info(f"Raw tool commands: {decision}")
            

            # with open("llm_decision_log.txt", "r") as f:
            #     decision = json.load(f)


            # Validate and correct tool commands
            if self.tool_validator and 'tool_commands' in decision:
                decision['tool_commands'] = self._validate_and_correct_commands(
                    decision['tool_commands']
                )
            
            # Add intelligent suggestions if no commands provided
            if not decision.get('tool_commands'):
                decision['tool_commands'] = self._generate_intelligent_suggestions(
                    error_patterns, failure_analysis, self.state.retrieval_history
                )
            
            # Update state
            self._update_state(decision)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in decide_context_update: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._get_intelligent_fallback(error_patterns, failure_analysis)
    
    def execute_context_decisions(self, current_context: Dict, 
                                 decision: Dict,
                                 tools: Dict,
                                 bug_info: Dict) -> Dict:
        """
        Execute validated tool commands and build dynamic context.
        
        Args:
            decision: Decision dict with tool commands
            tools: Available tool instances
            bug_info: Bug information for context
            
        Returns:
            Dynamic context with retrieved information
        """
        dynamic_context = {}
        execution_results = []
        
        for cmd in decision.get('tool_commands', []):
            tool_name = cmd['tool']
            params = cmd['params']
            confidence = cmd.get('confidence', 1.0)
            purpose = cmd.get('purpose', '')
            
            self.logger.info(f"Executing {tool_name} (confidence: {confidence:.2f}) - {purpose}")
            
            # Skip very low confidence unless critical
            if confidence < 0.3 and not self._is_critical_tool(tool_name):
                self.logger.warning(f"Skipping very low confidence tool: {tool_name}")
                continue
            
            # Check cache
            cache_key = self._get_cache_key(tool_name, params)
            if cache_key in self.tool_cache:
                self.logger.debug(f"Using cached result for {tool_name}")
                result = self.tool_cache[cache_key]
                execution_results.append({'tool': tool_name, 'status': 'cached', 'result': result})
            else:
                # Execute tool
                tool = tools.get(tool_name)
                if not tool:
                    self.logger.error(f"Tool not found: {tool_name}")
                    self.state.failed_tool_attempts.append({
                        'tool': tool_name,
                        'reason': 'Tool not found',
                        'iteration': self.state.iteration
                    })
                    continue
                
                try:
                    result = self._execute_tool_safely(tool, tool_name, params, bug_info)
                    
                    if result is not None:
                        # Cache successful result
                        self.tool_cache[cache_key] = result
                        execution_results.append({'tool': tool_name, 'status': 'success', 'result': result})
                        self.state.successful_retrievals.append({
                            'tool': tool_name,
                            'params': params,
                            'iteration': self.state.iteration
                        })
                    else:
                        # Try fallback
                        fallback_result = self._try_fallback_strategy(tool_name, params, tools, bug_info)
                        if fallback_result:
                            result = fallback_result
                            execution_results.append({'tool': tool_name, 'status': 'fallback', 'result': result})
                    
                except Exception as e:
                    self.logger.error(f"Error executing {tool_name}: {e}")
                    self.state.failed_tool_attempts.append({
                        'tool': tool_name,
                        'error': str(e),
                        'iteration': self.state.iteration
                    })
                    # Try fallback
                    result = self._try_fallback_strategy(tool_name, params, tools, bug_info)
                    if result:
                        execution_results.append({'tool': tool_name, 'status': 'fallback', 'result': result})
            
            # Add to context if we have a result
            if result is not None:
                context_key = self._get_semantic_context_key(tool_name, params)
                dynamic_context[context_key] = {
                    'data': result,
                    'tool': tool_name,
                    'params': params,
                    'confidence': confidence,
                    'purpose': purpose,
                    'timestamp': time.time()
                }
                
                # Update tool effectiveness
                self._update_tool_effectiveness(tool_name, result)
        
        # Apply context updates from decision
        if 'context_updates' in decision:
            dynamic_context = self._apply_context_updates(
                current_context,
                dynamic_context,
                decision['context_updates']
            )
        
        # Merge with previous successful retrievals if needed
        dynamic_context = self._merge_with_successful_history(dynamic_context)
        
        # Optimize for token limit
        optimized_context = self._optimize_for_tokens(dynamic_context)
        
        # Log execution summary
        self.logger.info(f"Context update complete: {len(execution_results)} tools executed, "
                        f"{len(optimized_context)} context items retained")
        
        return optimized_context
    
    def _execute_tool_safely(self, tool, tool_name: str, params: Dict, bug_info: Dict) -> Any:
        """
        Execute tool with proper parameter handling for each tool type.
        """
        self.logger.debug(f"Executing {tool_name} with params: {json.dumps(params, indent=2)}")
        
        try:
            if tool_name == 'similar_method_search':
                # Get method body from params or bug info
                method_body = params.get('method_body', '')
                if not method_body and bug_info.get('buggy_method'):
                    method_body = bug_info['buggy_method']
                return retrieve_context(bug_info.get('bug_id', 'unknown'), [method_body])
                # return tool.search(
                #     method_body=method_body,
                #     top_k=params.get('top_k', 5),
                #     filter_keywords=params.get('filter_keywords', [])
                # )
            
            elif tool_name == 'code_extractor':
                # Enhanced code extractor handles all extraction
                result = tool.extract(
                    file_path=params.get('file_path', ''),
                    element_name=params.get('element_name', ''),
                    element_type=params.get('element_type', 'method'),
                    language=params.get('language', None)
                )
                
                # Format result based on element type
                if params.get('element_type') == 'test' and result:
                    # Return structured test data
                    return {
                        'code': result.get('code', ''),
                        'assertions': result.get('assertions', []),
                        'setup': result.get('setup', ''),
                        'name': params.get('element_name', ''),
                        'type': 'test'
                    }
                
                return result
            
            elif tool_name == 'field_dependency_analyzer':
                return tool.analyze_fields(
                    class_name=params.get('class_name', ''),
                    method_name=params.get('method_name', ''),
                    file_path=params.get('file_path', None)
                )
            
            elif tool_name == 'call_graph_builder':
                result = tool.build_call_graph(
                    method_name=params.get('method_name', ''),
                    file_path=params.get('file_path', None),
                    direction=params.get('direction', 'both'),
                    max_depth=params.get('max_depth', 2)
                )
                
                # Return based on requested direction
                if params.get('direction') == 'callers':
                    return {'callers': result.get('callers', [])}
                elif params.get('direction') == 'callees':
                    return {'callees': result.get('callees', [])}
                return result
            
            elif tool_name == 'api_usage_finder':
                return tool.find_usage(
                    api_class=params.get('api_class', ''),
                    api_method=params.get('api_method', ''),
                    max_examples=params.get('max_examples', 5)
                )
            
            elif tool_name == 'coverage_runner':
                return tool.get_test_coverage(
                    test_methods=params.get('test_methods', [])
                )
            
            else:
                self.logger.warning(f"Unknown tool: {tool_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in tool execution for {tool_name}: {e}")
            raise
    
    def _validate_and_correct_commands(self, commands: List[Dict]) -> List[Dict]:
        """
        Validate and intelligently correct tool commands.
        """
        validated_commands = []
        
        for cmd in commands:
            tool_name = cmd.get('tool', '')
            params = cmd.get('params', {})
            purpose = cmd.get('purpose', '')
            
            # Handle legacy tool names
            if tool_name == 'test_code_extractor':
                self.logger.info("Converting test_code_extractor to code_extractor")
                tool_name = 'code_extractor'
                if 'test_name' in params:
                    params['element_name'] = params.pop('test_name')
                if 'test_path' in params:
                    params['file_path'] = params.pop('test_path', '')
                params['element_type'] = 'test'
            
            # Validate with tool validator
            if self.tool_validator:
                resolved = self.tool_validator.validate_and_resolve(tool_name, params)
                
                validated_cmd = {
                    'tool': resolved.tool_name,
                    'params': resolved.params,
                    'purpose': purpose or f"Retrieve {resolved.tool_name} information",
                    'confidence': resolved.confidence,
                    'resolution_notes': resolved.resolution_notes
                }
                
                if resolved.resolution_notes:
                    self.logger.info(f"Tool resolution notes: {resolved.resolution_notes}")
            else:
                # Basic validation without validator
                validated_cmd = {
                    'tool': tool_name,
                    'params': params,
                    'purpose': purpose,
                    'confidence': 0.8
                }
            
            validated_commands.append(validated_cmd)
        
        return validated_commands
    
    def _get_enhanced_system_prompt(self) -> str:
        """Get comprehensive system prompt with all tool details."""
        return """You are an intelligent context manager for automated bug fixing.
Your role is to analyze failed patch attempts and decide what additional information to retrieve.

AVAILABLE TOOLS (with exact parameters):

1. code_extractor - Extract any code element (HANDLES TESTS TOO!)
   Parameters:
   - element_name: str (name of element to extract)
   - element_type: str ('method', 'class', 'field', 'test')
   - file_path: str (optional, will be resolved if not provided)
   - language: str (optional, auto-detected)
   
   For extracting test code, use element_type='test'
   This will extract test code with assertions and setup methods

2. similar_method_search - Find similar method implementations
   Parameters:
   - method_body: str (the buggy method code) SHOULD BE FULL BODY CODE (not part of th method! and not '...' mark!)
   - top_k: int (number of results, default 5)
   - filter_keywords: list (optional keywords to filter)

3. field_dependency_analyzer - Analyze field usage in methods
   Parameters:
   - class_name: str (name of the class)
   - method_name: str (name of the method)
   - file_path: str (optional)

4. call_graph_builder - Build call relationships
   Parameters:
   - method_name: str (target method)
   - file_path: str (optional)
   - direction: str ('callers', 'callees', or 'both')
   - max_depth: int (traversal depth, default 2)

5. api_usage_finder - Find API usage examples
   Parameters:
   - api_class: str (API class name)
   - api_method: str (API method name)
   - max_examples: int (default 5)

6. coverage_runner - Get executed methods when a specific test is run
   Parameters:
   - test_methods: list of test method names (The test name should be full. For exammple: org.jfree.chart.util.junit.ShapeUtilitiesTests::testEqualGeneralPaths)


IMPORTANT NOTE:

- Avoid repeating the same tool with identical parameters

Return a JSON response with this structure:
{
    "tool_commands": [
        {
            "tool": "tool_name",
            "params": {
                "param1": "value1",
                "param2": "value2"
            },
            "purpose": "Why this tool helps address the current failure"
        }
    ],
    "context_updates": {
        "keep": ["list of context keys to retain"],
        "remove": ["list of context keys to remove"],
        "add": {}
    },
    "reasoning": "Overall strategy explanation",
    "focus": "What specific aspect to focus on next"
}

"""
    
    def _build_decision_prompt(self, buggy_method, kb, tried_hypotheses: List[Dict],
                              current_context: Dict, retrieval_history: List[Dict],
                              error_patterns: Dict, failure_analysis: Dict) -> str:
        """Build comprehensive prompt for decision making."""
        prompt = f"""## Iteration {self.state.iteration} - Context Update Decision
### ORIGINAL BUGGY METHOD:
```java
{buggy_method if buggy_method else 'Not provided'}
```"""
        prompt+= f"""### FAILING TESTS ({len(kb['failing_tests'])} tests):
"""      
        # Add test information
        for i, test in enumerate(kb['failing_tests'], start=1):
            if i > 10:
                break
            test_key = f'test_code_{test}'
            if test_key in kb:
                test_info = kb[test_key]
                prompt += f"""
#### Test: {test}:
```java    
{test_info['code'] if 'code' in test_info else ''}
```
"""   
        prompt += f"""
### ERROR PATTERNS DETECTED IN THE PREVIOUS ATTEMPTS:
"""
        # Comprehensive error analysis
        if error_patterns.get('error_types'):
            prompt += "Error Type Distribution:\n"
            for error_type, count in sorted(error_patterns['error_types'].items(), 
                                        key=lambda x: x[1], reverse=True):
                prompt += f"  - {error_type}: {count} occurrences\n"

        # Add test-specific patterns
        if error_patterns.get('test_specific_patterns'):
            prompt += "\nPer-Test Error Patterns:\n"
            for test, errors in list(error_patterns['test_specific_patterns'].items())[:5]:
                error_summary = ', '.join([f"{e}({c})" for e, c in errors.items()])
                prompt += f"  - {test}: {error_summary}\n"

        # Add persistent test failures
        if error_patterns.get('persistent_tests'):
            prompt += f"\nTests failing in ALL attempts: {', '.join(error_patterns['persistent_tests'][:5])}\n"
        
        prompt += f"""

### RECENT ATTEMPTS ({len(tried_hypotheses)} total):
"""
        # Show last 3 attempts with details
        for i, hyp in enumerate(tried_hypotheses[-5:], 1):
            result = hyp.get('execution_result', {})
            prompt += f"""
Attempt {i}:
- Hypothesis: {hyp.get('hypothesis', 'N/A')}
- Changes: {hyp.get('changes', 'N/A')}
- Result: 
    - Still failing tests : {result.get('failed_tests', '')}
    - Passed tests : {result.get('passed_tests', '')}
"""

        # Add current context summary
        prompt += f"""

### CURRENT CONTEXT:
Items: {len(current_context)}
Keys: {list(current_context.keys())[:10]}
Token usage: ~{self._estimate_tokens(current_context)}/{self.max_tokens}

### PREVIOUS RETRIEVALS:
"""
        # Show retrieval history
        tool_counts = defaultdict(int)
        for item in retrieval_history:
            tool_counts[item.get('tool', 'Unknown')] += 1
        
        for tool, count in tool_counts.items():
            prompt += f"- {tool}: used {count} times\n"

        # Add specific guidance based on patterns
        prompt += """
"""
        prompt += """

Based on the error patterns and failed attempts, decide what NEW information would be most helpful.
Avoid repeating previous retrievals unless with different parameters.
Focus on addressing the root cause of the failures.
"""
        
        return prompt
    
    def _analyze_error_patterns(self, tried_hypotheses: List[Dict]) -> Dict:
        """Enhanced error pattern analysis for multiple test failures."""
        patterns = {
            'error_types': defaultdict(int),
            'error_locations': defaultdict(list),
            'test_failures': defaultdict(list),
            'persistent_tests': set(),
            'error_evolution': [],
            'test_specific_patterns': defaultdict(lambda: defaultdict(int))
        }
        
        # Analyze each hypothesis attempt
        for i, hyp in enumerate(tried_hypotheses):
            result = hyp.get('execution_result', {})
            
            # Handle multiple error details
            error_details = result.get('error_details', {})
            if error_details:
                # Detailed analysis for multiple errors
                iteration_errors = {
                    'iteration': i + 1,
                    'test_errors': {}
                }
                
                for test_name, error_info in error_details.items():
                    error_type = error_info.get('error_type', 'Unknown')
                    error_line = error_info.get('error_line')
                    
                    # Track patterns
                    patterns['error_types'][error_type] += 1
                    patterns['test_failures'][test_name].append(error_type)
                    patterns['test_specific_patterns'][test_name][error_type] += 1
                    
                    if error_line:
                        patterns['error_locations'][error_type].append(error_line)
                    
                    iteration_errors['test_errors'][test_name] = error_type
                
                patterns['error_evolution'].append(iteration_errors)
                
                # Track persistent test failures
                if i == 0:
                    patterns['persistent_tests'] = set(error_details.keys())
                else:
                    patterns['persistent_tests'] &= set(error_details.keys())
            
            else:
                # Fallback to single error (backward compatibility)
                error_type = result.get('error_type', 'Unknown')
                patterns['error_types'][error_type] += 1
        
        patterns['persistent_tests'] = list(patterns['persistent_tests'])
        
        return patterns
    
    def _analyze_failure_reasons(self, tried_hypotheses: List[Dict]) -> Dict:
        """Analyze failure reasons to identify patterns."""
        analysis = {
            'common_failure_reason': None,
            'unique_approaches_tried': set(),
            'successful_partial_fixes': []
        }
        
        failure_reasons = []
        for hyp in tried_hypotheses:
            if 'failure_reason' in hyp:
                failure_reasons.append(hyp['failure_reason'])
            if 'approach_type' in hyp:
                analysis['unique_approaches_tried'].add(hyp['approach_type'])
            
            # Check for partial successes
            result = hyp.get('execution_result', {})
            if result.get('tests_passed', 0) > 0:
                analysis['successful_partial_fixes'].append({
                    'hypothesis': hyp.get('hypothesis'),
                    'tests_passed': result['tests_passed']
                })
        
        # Find most common failure reason
        if failure_reasons:
            from collections import Counter
            reason_counts = Counter(failure_reasons)
            analysis['common_failure_reason'] = reason_counts.most_common(1)[0][0]
        
        return analysis
    
    def _generate_intelligent_suggestions(self, error_patterns: Dict, 
                                         failure_analysis: Dict,
                                         retrieval_history: List[Dict]) -> List[Dict]:
        """Generate intelligent tool suggestions based on analysis."""
        suggestions = []
        used_tools = set(item.get('tool') for item in retrieval_history)
        
        # Get most common error
        if error_patterns.get('error_types'):
            most_common_error = max(error_patterns['error_types'].items(), key=lambda x: x[1])[0]
            
            # Get suggested tools for this error
            suggested_tools = self.error_to_tool_mapping.get(most_common_error, [])
            
            for tool in suggested_tools:
                if tool not in used_tools or self.state.iteration > 5:
                    if tool == 'code_extractor':
                        # Decide what to extract based on error
                        if 'Assertion' in most_common_error:
                            suggestions.append({
                                'tool': 'code_extractor',
                                'params': {'element_type': 'test'},
                                'purpose': f"Extract test to understand assertion failure"
                            })
                        else:
                            suggestions.append({
                                'tool': 'code_extractor',
                                'params': {'element_type': 'method'},
                                'purpose': f"Extract related method code"
                            })
                    
                    elif tool == 'similar_method_search':
                        suggestions.append({
                            'tool': 'similar_method_search',
                            'params': {'top_k': 5},
                            'purpose': f"Find methods that handle {most_common_error}"
                        })
                    
                    elif tool == 'field_dependency_analyzer':
                        suggestions.append({
                            'tool': 'field_dependency_analyzer',
                            'params': {},
                            'purpose': f"Analyze field dependencies for {most_common_error}"
                        })
                    
                    if len(suggestions) >= 3:
                        break
        
        # Add fallback suggestion if no specific suggestions
        if not suggestions:
            suggestions.append({
                'tool': 'similar_method_search',
                'params': {'top_k': 5},
                'purpose': "Find similar working implementations"
            })
        
        return suggestions
    
    def _initialize_error_mapping(self) -> Dict[str, List[str]]:
        """Initialize error type to tool mapping."""
        return {
            'NullPointerException': ['field_dependency_analyzer', 'similar_method_search', 'code_extractor'],
            'IndexOutOfBoundsException': ['similar_method_search', 'code_extractor', 'api_usage_finder'],
            'ArrayIndexOutOfBoundsException': ['similar_method_search', 'code_extractor'],
            'ClassCastException': ['code_extractor', 'field_dependency_analyzer', 'api_usage_finder'],
            'AssertionError': ['code_extractor', 'similar_method_search', 'call_graph_builder'],
            'AssertionFailedError': ['code_extractor', 'similar_method_search'],
            'CompilationError': ['code_extractor', 'call_graph_builder', 'api_usage_finder'],
            'IllegalArgumentException': ['api_usage_finder', 'similar_method_search', 'code_extractor'],
            'IllegalStateException': ['field_dependency_analyzer', 'call_graph_builder'],
            'ArithmeticException': ['similar_method_search', 'code_extractor'],
            'NumberFormatException': ['api_usage_finder', 'similar_method_search'],
            'Unknown': ['similar_method_search', 'code_extractor']
        }
    
    def _get_intelligent_fallback(self, error_patterns: Dict, failure_analysis: Dict) -> Dict:
        """Get intelligent fallback decision based on analysis."""
        # Generate suggestions based on patterns
        suggestions = self._generate_intelligent_suggestions(
            error_patterns, failure_analysis, self.state.retrieval_history
        )
        
        return {
            "tool_commands": suggestions[:2],  # Limit to 2 tools
            "context_updates": {"keep": [], "remove": [], "add": {}},
            "reasoning": "Fallback decision based on error pattern analysis",
            "focus": f"Address {list(error_patterns.get('error_types', {}).keys())[0] if error_patterns.get('error_types') else 'Unknown error'}"
        }
    
    def _try_fallback_strategy(self, tool_name: str, params: Dict, 
                              tools: Dict, bug_info: Dict) -> Optional[Any]:
        """Try fallback strategies when tool execution fails."""
        self.logger.info(f"Attempting fallback for {tool_name}")
        
        fallback_strategies = {
            'code_extractor': lambda: self._fallback_code_extraction(params, bug_info),
            'similar_method_search': lambda: self._fallback_similar_search(params, bug_info),
            'field_dependency_analyzer': lambda: self._fallback_field_analysis(params, bug_info),
            'call_graph_builder': lambda: self._fallback_call_graph(params, bug_info)
        }
        
        if tool_name in fallback_strategies:
            try:
                result = fallback_strategies[tool_name]()
                if result:
                    self.logger.info(f"Fallback successful for {tool_name}")
                    return result
            except Exception as e:
                self.logger.error(f"Fallback also failed for {tool_name}: {e}")
        
        return None
    
    def _fallback_code_extraction(self, params: Dict, bug_info: Dict) -> Dict:
        """Fallback for code extraction."""
        element_type = params.get('element_type', 'method')
        element_name = params.get('element_name', '')
        
        if element_type == 'test':
            # Return placeholder test structure
            return {
                'code': f"// Test code not available for {element_name}",
                'assertions': [],
                'setup': '',
                'error': 'Extraction failed - using placeholder'
            }
        else:
            # Try to get from bug info
            if element_type == 'method' and bug_info.get('buggy_method'):
                return {
                    'code': bug_info['buggy_method'],
                    'type': 'method',
                    'source': 'bug_info'
                }
        
        return {'code': '', 'error': 'Extraction failed'}
    
    def _fallback_similar_search(self, params: Dict, bug_info: Dict) -> List:
        """Fallback for similar method search."""
        # Return empty list but valid structure
        return []
    
    def _fallback_field_analysis(self, params: Dict, bug_info: Dict) -> List:
        """Fallback for field dependency analysis."""
        return []
    
    def _fallback_call_graph(self, params: Dict, bug_info: Dict) -> Dict:
        """Fallback for call graph builder."""
        return {'callers': [], 'callees': []}
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        """Parse LLM response with multiple strategies."""
        try:
            # Direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Try finding JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        # Return minimal valid structure
        self.logger.warning("Could not parse LLM response, using minimal structure")
        return {
            "tool_commands": [],
            "context_updates": {"keep": [], "remove": [], "add": {}},
            "reasoning": "Could not parse response",
            "focus": "Continue with existing context"
        }
    
    def _update_state(self, decision: Dict):
        """Update internal state based on decision."""
        # Track tool usage
        for cmd in decision.get('tool_commands', []):
            self.state.retrieval_history.append({
                'iteration': self.state.iteration,
                'tool': cmd['tool'],
                'params': cmd['params'],
                'confidence': cmd.get('confidence', 1.0),
                'purpose': cmd.get('purpose', '')
            })
        
        # Update token count estimate
        self.state.token_count = self._estimate_tokens(self.state.retrieved_data)
    
    def _apply_context_updates(self, current: Dict, retrieved: Dict, updates: Dict) -> Dict:
        """Apply context update instructions from LLM."""
        result = {}
        
        # Keep specified keys
        for key in updates.get('keep', []):
            if key in current:
                result[key] = current[key]
            elif key in retrieved:
                result[key] = retrieved[key]
        
        # Remove specified keys (by not adding them)
        remove_keys = set(updates.get('remove', []))
        
        # Add all current items not marked for removal
        for key, value in current.items():
            if key not in remove_keys and key not in result:
                result[key] = value

        for key, value in retrieved.items():
            if key not in remove_keys and key not in result:
                result[key] = value
                
        # Add new items
        for key, value in updates.get('add', {}).items():
            result[key] = value
        
        return result
    
    def _merge_with_successful_history(self, dynamic_context: Dict) -> Dict:
        """Merge current context with successful historical retrievals."""
        # Keep track of successful patterns
        if self.state.successful_retrievals:
            # Add successful patterns that aren't in current context
            for retrieval in self.state.successful_retrievals[-3:]:  # Last 3 successful
                key = self._get_semantic_context_key(
                    retrieval['tool'],
                    retrieval.get('params', {})
                )
                if key not in dynamic_context and key in self.state.retrieved_data:
                    dynamic_context[f"historical_{key}"] = self.state.retrieved_data[key]
        
        return dynamic_context
    
    def _optimize_for_tokens(self, context: Dict) -> Dict:
        """Optimize context to fit within token budget."""
        current_tokens = self._estimate_tokens(context)
        
        if current_tokens <= self.max_tokens:
            return context
        
        self.logger.info(f"Optimizing context: {current_tokens} -> {self.max_tokens} tokens")
        
        # Prioritize context items
        prioritized = []
        for key, value in context.items():
            priority = self._calculate_priority(key, value)
            item_tokens = self._estimate_tokens(value)
            prioritized.append((priority, key, value, item_tokens))
        
        # Sort by priority (highest first)
        prioritized.sort(key=lambda x: x[0], reverse=True)
        
        # Build optimized context
        optimized = {}
        total_tokens = 0
        
        for priority, key, value, item_tokens in prioritized:
            if total_tokens + item_tokens <= self.max_tokens:
                optimized[key] = value
                total_tokens += item_tokens
            elif total_tokens + 100 <= self.max_tokens:
                # Try to add summary
                summary = self._create_summary(key, value)
                summary_tokens = self._estimate_tokens(summary)
                if total_tokens + summary_tokens <= self.max_tokens:
                    optimized[f"{key}_summary"] = summary
                    total_tokens += summary_tokens
        
        self.logger.info(f"Optimized to {total_tokens} tokens, {len(optimized)} items")
        return optimized
    
    def _calculate_priority(self, key: str, value: Any) -> float:
        """Calculate priority score for context item."""
        priority = 0.5
        
        # Higher priority for test code
        if 'test' in key.lower():
            priority += 0.3
        
        # Higher priority for recent items
        if isinstance(value, dict) and 'timestamp' in value:
            age = time.time() - value['timestamp']
            recency_score = max(0, 1 - age / 3600)  # Decay over 1 hour
            priority += recency_score * 0.2
        
        # Higher priority for high confidence items
        if isinstance(value, dict) and 'confidence' in value:
            priority += value['confidence'] * 0.2
        
        # Higher priority for error-specific context
        if isinstance(value, dict) and 'purpose' in value:
            if 'error' in value['purpose'].lower() or 'assertion' in value['purpose'].lower():
                priority += 0.2
        
        return min(1.0, priority)
    
    def _create_summary(self, key: str, value: Any) -> str:
        """Create concise summary of context value."""
        if isinstance(value, dict):
            if 'data' in value:
                data = value['data']
                if isinstance(data, list):
                    return f"[{len(data)} items from {value.get('tool', 'unknown')}]"
                elif isinstance(data, dict):
                    return f"[{value.get('tool', 'unknown')} result with {len(data)} keys]"
                else:
                    return str(data)[:200]
            else:
                return f"[{key}: {len(value)} fields]"
        elif isinstance(value, list):
            return f"[List of {len(value)} items]"
        else:
            return str(value)[:200]
    
    def _update_tool_effectiveness(self, tool_name: str, result: Any):
        """Track tool effectiveness for learning."""
        effectiveness = 0.0
        
        if result:
            if isinstance(result, list):
                effectiveness = min(1.0, len(result) / 5.0) if result else 0.0
            elif isinstance(result, dict):
                if result.get('code'):
                    effectiveness = 0.8 if len(result['code']) > 100 else 0.4
                elif result.get('data'):
                    effectiveness = 0.7
                else:
                    effectiveness = 0.5 if len(result) > 0 else 0.0
            else:
                effectiveness = 0.5
        
        # Update running average
        current = self.state.tool_effectiveness.get(tool_name, 0.5)
        self.state.tool_effectiveness[tool_name] = 0.7 * current + 0.3 * effectiveness
        
        self.logger.debug(f"Tool {tool_name} effectiveness: {self.state.tool_effectiveness[tool_name]:.2f}")
    
    def _is_critical_tool(self, tool_name: str) -> bool:
        """Determine if tool is critical for bug fixing."""
        critical_tools = ['code_extractor', 'similar_method_search']
        return tool_name in critical_tools
    
    def _get_semantic_context_key(self, tool_name: str, params: Dict) -> str:
        """Generate semantic key for context item."""
        if tool_name == 'code_extractor':
            element_type = params.get('element_type', 'method')
            element_name = params.get('element_name', 'unknown')
            if element_type == 'test':
                return f"test_code_{element_name}"
            return f"extracted_{element_type}_{element_name}"
        
        elif tool_name == 'similar_method_search':
            return f"similar_methods_top_{params.get('top_k', 5)}"
        
        elif tool_name == 'field_dependency_analyzer':
            return f"field_dependencies_{params.get('class_name', 'unknown')}_{params.get('method_name', 'unknown')}"
        
        elif tool_name == 'call_graph_builder':
            return f"call_graph_{params.get('method_name', 'unknown')}_{params.get('direction', 'both')}"
        
        elif tool_name == 'api_usage_finder':
            return f"api_usage_{params.get('api_class', 'unknown')}_{params.get('api_method', 'unknown')}"
        
        elif tool_name == 'coverage_runner':
            return "test_coverage"
        
        return f"{tool_name}_result"
    
    def _get_cache_key(self, tool_name: str, params: Dict) -> str:
        """Generate cache key for tool results."""
        # Create deterministic key from tool and params
        key_data = {
            'tool': tool_name,
            'params': params
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _estimate_tokens(self, data: Any) -> int:
        """Estimate token count for data."""
        if data is None:
            return 0
        
        text = json.dumps(data) if not isinstance(data, str) else data
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    def get_state_summary(self) -> Dict:
        """Get summary of current state for monitoring."""
        return {
            'iteration': self.state.iteration,
            'token_count': self.state.token_count,
            'retrievals_made': len(self.state.retrieval_history),
            'successful_retrievals': len(self.state.successful_retrievals),
            'failed_attempts': len(self.state.failed_tool_attempts),
            'tool_effectiveness': dict(self.state.tool_effectiveness),
            'cache_size': len(self.tool_cache),
            'confidence_avg': sum(self.state.confidence_scores) / len(self.state.confidence_scores) if self.state.confidence_scores else 0
        }