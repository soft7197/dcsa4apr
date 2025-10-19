import openai
from typing import Dict, List, Tuple, Optional
import difflib
import json

class PromptMakerAgent:
    def __init__(self):
        self.template_cache = {}
        
    def create_prompt(self,
                     knowledge_base: Dict,
                     dynamic_context: Dict,
                     tried_hypotheses: List[Dict]) -> str:
        """Create optimized prompt for patch generation."""
        
        prompt = f"""## Bug Fixing Task

### Buggy Method:
```{knowledge_base.get('language', 'java')}{knowledge_base['buggy_method']}

### Error Information:
- Failing Tests: {knowledge_base.get('failing_tests', [])}
"""
        
        # Add detailed error messages
        for error in knowledge_base.get('error_messages', []):
            prompt += f"""
Test: {error['test']}
Error Type: {error['error_type']}
Error Message: {error['error_message']}
Stack Trace (relevant part):
{error.get('stack_trace', 'N/A')[:500]}
"""

        # Add failing test source code
        prompt += "\n### Failing Test Code:\n"
        for test_detail in knowledge_base.get('failing_tests_details', []):
            test_name = test_detail.get('test_name', 'Unknown')
            test_src = test_detail.get('src', test_detail.get('test_source', ''))
            
            if test_src:
                prompt += f"""
#### Test: {test_name}
```{knowledge_base.get('language', 'java')}{test_src}
"""
            else:
                # Try to get from dynamic context if available
                test_code = dynamic_context.get(f'test_code_{test_name}', '')
                if test_code:
                    prompt += f"""
#### Test: {test_name}
```{knowledge_base.get('language', 'java')}{test_code}
"""

        # Add issue information if available
        if 'issue_title' in knowledge_base:
            prompt += f"""
### Issue Information:
Title: {knowledge_base.get('issue_title', '')}
Description: {knowledge_base.get('issue_description', '')}
"""

        # Add previous failed attempts
        if tried_hypotheses:
            prompt += "\n### Previous Failed Attempts:\n"
            for i, hyp in enumerate(tried_hypotheses[-3:], 1):
                prompt += f"""
Attempt {i}:
- Hypothesis: {hyp['summary']}
- Changes Made: {hyp['changes']}
- Why it failed: {hyp.get('failure_reason', 'Unknown')}
"""

        # Add ALL dynamic context information
        prompt += "\n### Retrieved Context Information:\n"
        
        # Similar methods
        if 'similar_methods' in dynamic_context:
            prompt += "\n#### Similar Correct Implementations:\n"
            for method in dynamic_context['similar_methods'][:3]:
                prompt += f"""
```{knowledge_base.get('language', 'java')}{method['code']}
Similarity Score: {method['similarity']:.3f}
"""
        
        # Extracted code from other files
        if 'extracted_code' in dynamic_context:
            prompt += f"\n#### Related Code Extracted:\n"
            for code_item in dynamic_context.get('extracted_code', []):
                prompt += f"""
File: {code_item.get('file', 'Unknown')}
```{knowledge_base.get('language', 'java')}{code_item.get('code', '')}
"""
        
        # Call graph information
        if 'call_graph' in dynamic_context:
            call_info = dynamic_context['call_graph']
            prompt += f"\n#### Call Graph Information:\n"
            
            if 'callers' in call_info and call_info['callers']:
                prompt += "Methods that call this buggy method:\n"
                for caller in call_info['callers'][:5]:
                    prompt += f"  - {caller.get('class', '')}.{caller.get('method', '')}\n"
            
            if 'callees' in call_info and call_info['callees']:
                prompt += "Methods called by this buggy method:\n"
                for callee in call_info['callees'][:5]:
                    prompt += f"  - {callee.get('class', '')}.{callee.get('method', '')}\n"
        
        # Field dependencies
        if 'field_dependencies' in dynamic_context:
            prompt += f"\n#### Field Dependencies:\n"
            for field in dynamic_context['field_dependencies']:
                prompt += f"  - {field.get('type', '')} {field.get('name', '')}: {field.get('usage', '')}\n"
        
        # Related test cases that pass
        if 'passing_tests' in dynamic_context:
            prompt += f"\n#### Similar Passing Tests (for reference):\n"
            for test in dynamic_context['passing_tests'][:2]:
                prompt += f"""
Test: {test.get('name', '')}
```{knowledge_base.get('language', 'java')}{test.get('code', '')}
"""
        
        # API usage examples
        if 'api_usage_examples' in dynamic_context:
            prompt += f"\n#### API Usage Examples:\n"
            for example in dynamic_context['api_usage_examples'][:3]:
                prompt += f"""
```{knowledge_base.get('language', 'java')}{example}
"""
        
        # Any other retrieved information
        for key, value in dynamic_context.items():
            if key not in ['similar_methods', 'extracted_code', 'call_graph', 
                          'field_dependencies', 'passing_tests', 'api_usage_examples'] \
               and not key.startswith('test_code_'):
                prompt += f"\n#### {key.replace('_', ' ').title()}:\n"
                if isinstance(value, str):
                    prompt += f"{value}\n"
                elif isinstance(value, list):
                    for item in value[:5]:  # Limit to first 5 items
                        prompt += f"  - {item}\n"
                elif isinstance(value, dict):
                    for k, v in list(value.items())[:5]:  # Limit to first 5 items
                        prompt += f"  - {k}: {v}\n"

        prompt += """
### Task:
Based on the error, failing tests, and all context provided, generate a fix for the buggy method.

Provide your response in the following JSON format:
{
    "hypothesis": "Brief explanation of the bug cause and fix approach",
    "changes": "Description of specific changes in +/- format",
    "fixed_method": "Complete fixed method code without comments"
}
"""
        
        return prompt
    
    def create_multi_function_prompt(self,
                                    knowledge_base: Dict,
                                    dynamic_context: Dict,
                                    tried_hypotheses: List[Dict],
                                    is_current_method: bool = True) -> str:
        """Create prompt for multi-function bug fixing."""
        
        prompt = f"""## Multi-Function Bug Fixing Task

### Bug ID: {knowledge_base.get('bug_id')}
This bug requires fixing {len(knowledge_base['buggy_methods'])} methods together to pass the tests.

### All Buggy Methods:
"""
        
        # List all buggy methods
        for i, method in enumerate(knowledge_base['buggy_methods']):
            method_key = f"{method.class_name}.{method.method_name}"
            prompt += f"""
#### Method {i+1}: {method_key}
Location: {method.file_path}
Lines: {method.line_numbers[0]}-{method.line_numbers[-1]}
"""
            
            if f'buggy_fl_{method_key}' in knowledge_base:
                prompt += f"""
```java{knowledge_base[f'buggy_fl_{method_key}']}
"""
            else:
                prompt += f"""
```java{knowledge_base['buggy_codes'].get(method_key, '')}
"""
        
        # Add current method focus
        current_index = knowledge_base.get('current_method_index', 0)
        current_method = knowledge_base['buggy_methods'][current_index]
        current_key = f"{current_method.class_name}.{current_method.method_name}"
        
        prompt += f"""
### Current Method to Fix: {current_key}
{knowledge_base.get('current_method', '')}
"""

        # Add issue information
        if 'issue_title' in knowledge_base:
            prompt += f"""
### Issue Information:
Title: {knowledge_base['issue_title']}
Description: {knowledge_base['issue_description']}
"""
        
        # Add error information
        prompt += "\n### Error Information:\n"
        for error in knowledge_base.get('error_messages', []):
            prompt += f"""
Test: {error['test']}
Error Type: {error['error_type']}
Message: {error['error_message']}
"""

        # Add ALL failing test source code
        prompt += "\n### Failing Test Code:\n"
        for test_detail in knowledge_base.get('failing_tests_details', []):
            test_name = test_detail.get('test_name', 'Unknown')
            test_src = test_detail.get('src', test_detail.get('test_source', ''))
            
            if test_src:
                prompt += f"""
#### Test: {test_name}
```java{test_src}
"""
            # Also check dynamic context for test code
            elif f'test_code_{test_name}' in dynamic_context:
                prompt += f"""
#### Test: {test_name}
```java{dynamic_context[f'test_code_{test_name}']}
"""
        
        # Add previous attempts if any
        if tried_hypotheses:
            prompt += "\n### Previous Failed Attempts:\n"
            for i, hyp in enumerate(tried_hypotheses[-3:], 1):
                prompt += f"""
Attempt {i}:
- Methods Fixed: {hyp.get('methods_fixed', [])}
- Changes: {hyp.get('changes', 'N/A')}
- Failure: {hyp.get('execution_result', {}).get('error_message', 'Unknown')}
"""
        
        # Add ALL dynamic context
        prompt += "\n### Retrieved Context Information:\n"
        
        # Add all dynamic context as in single function prompt
        for key, value in dynamic_context.items():
            if not key.startswith('test_code_'):
                if key == 'similar_methods':
                    prompt += "\n#### Similar Correct Implementations:\n"
                    for method in value[:3]:
                        prompt += f"""
```java{method['code']}
Similarity: {method['similarity']:.3f}
"""
                elif key == 'call_graph':
                    prompt += "\n#### Call Graph Information:\n"
                    if 'callers' in value:
                        prompt += f"Callers: {', '.join([c.get('method', '') for c in value['callers'][:5]])}\n"
                    if 'callees' in value:
                        prompt += f"Callees: {', '.join([c.get('method', '') for c in value['callees'][:5]])}\n"
                else:
                    prompt += f"\n#### {key.replace('_', ' ').title()}:\n"
                    if isinstance(value, str):
                        prompt += f"{value}\n"
                    elif isinstance(value, (list, dict)):
                        prompt += f"{json.dumps(value, indent=2)[:1000]}\n"  # Limit size
        
        prompt += f"""
### Task:
Fix the current method ({current_key}) while being aware that all {len(knowledge_base['buggy_methods'])} methods need to work together.

IMPORTANT: 
- The fix must be coordinated with the other buggy methods
- All methods together should make the failing tests pass
- Consider the relationships between the methods
- Analyze the test code to understand what behavior is expected

Provide your response in JSON format:
{{
    "hypothesis": "Explanation of the bug and fix approach for this method",
    "changes": "Specific changes in +/- format",
    "fixed_method": "Complete fixed method code without comments",
    "coordination_notes": "How this fix coordinates with other methods"
}}
"""
        
        return prompt     

class GeneratorAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    def generate_patch(self, prompt: str, buggy_method, k) -> Dict:
        """Generate patch hypothesis and fixed code."""
        
        response = self.client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.7,
            n=k,
            response_format={"type": "json_object"}
        )
        
        results = []
        for choice in response.choices:
            patch = json.loads(choice.message.content)

            # Add unified diff format
            patch["diff"] = self._generate_diff(
                buggy_method,
                patch["fixed_method"]
            )

            results.append(patch)

        return results
    
    def generate_multi_patch(self, prompt, buggy_methods, k):
        """
        Generate patch hypothesis and fixed code for ALL buggy methods.
        
        Args:
            prompt (str): Repair prompt for the model.
            buggy_methods (List[Dict]): List of buggy methods with their file paths, etc.
        
        Returns:
            Dict: JSON with hypothesis, methods list, and coordination notes.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            n=k,
            response_format={"type": "json_object"}
        )
        
        all_results = []
        for choice in response.choices:
            result = json.loads(choice.message.content)

            final_result = {
                "hypothesis": result.get("hypothesis", ""),
                "changes": result.get("changes"),
                "methods": result.get("methods", [])
            }
            all_results.append(final_result)

        return all_results
    
    def _get_system_prompt(self) -> str:
        return """You are an expert software engineer specializing in automated program repair.
        Your task is to fix bugs in code based on test failures and error messages and the provided context.
        
        Guidelines:
        1. Analyze the error carefully to understand the root cause
        2. Learn from previous failed attempts - don't repeat the same mistakes
        3. Ensure the fixed code is syntactically correct
        4. Focus on the failing test requirements
        5. Use the provided context effectively
        """

    def _generate_diff(self, original: str, fixed: str) -> str:
        """Generate unified diff between original and fixed code."""
        original_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile='original',
            tofile='changed',
        )
        
        return ''.join(diff)

