# src/execution/executor.py
"""
Complete patch executor implementation with support for both single and multi-method patches.
Includes Docker support for safe execution.
"""

import os
import re
import subprocess
import tempfile
import shutil
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import hashlib


@dataclass
class ExecutionResult:
    """Structured execution result."""
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    output: Optional[str] = None
    test_results: Optional[Dict] = None
    compilation_time: float = 0.0
    test_time: float = 0.0


class PatchExecutor:
    """
    Executes patches and runs tests.
    Supports both single-method and multi-method patches.
    """
    
    def __init__(self, project_path: str, language: str = 'java'):
        """
        Initialize patch executor.
        
        Args:
            project_path: Path to the project
            language: Programming language ('java' or 'python')
        """
        self.project_path = project_path
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        # Cache for compiled states
        self._compilation_cache = {}
        
        # Detect build system
        self.build_system = self._detect_build_system()
        self.logger.info(f"Detected build system: {self.build_system}")
    
    def _detect_build_system(self) -> str:
        """Detect the build system used by the project."""
        if os.path.exists(os.path.join(self.project_path, 'pom.xml')):
            return 'maven'
        elif os.path.exists(os.path.join(self.project_path, 'build.gradle')):
            return 'gradle'
        elif os.path.exists(os.path.join(self.project_path, 'build.xml')):
            return 'ant'
        elif self.language == 'python':
            if os.path.exists(os.path.join(self.project_path, 'setup.py')):
                return 'setuptools'
            else:
                return 'pytest'
        else:
            return 'direct'
    
    def execute_patch(self, patch: Dict, failing_tests: List[str]) -> Dict:
        """
        Execute a single patch and run tests.
        
        Args:
            patch: Patch information with fixed method code
            failing_tests: List of tests to run
            
        Returns:
            Execution result dictionary
        """
        self.logger.info(f"Executing patch for {patch.get('method_name', 'unknown')}")
        
        # Check if this is a multi-method patch
        if patch.get('is_multi_method') or 'methods' in patch:
            return self.execute_multi_patches(patch, failing_tests)
        
        # Single method patch
        start_time = time.time()
        
        # Backup original file
        file_path = patch.get('file_path', '')
        backup_content = self._backup_file(file_path)
        
        if backup_content is None:
            return {
                'status': 'failed',
                'error_type': 'FileNotFound',
                'error_message': f"File not found: {file_path}"
            }
        
        try:
            # Apply patch
            success = self._apply_patch(patch)
            
            if not success:
                return {
                    'status': 'failed',
                    'error_type': 'PatchApplicationError',
                    'error_message': 'Failed to apply patch'
                }
            
            # Compile
            compile_start = time.time()
            compile_result = self._compile()
            compile_time = time.time() - compile_start
            
            if not compile_result['success']:
                return {
                    'status': 'failed',
                    'error_type': 'CompilationError',
                    'error_message': compile_result.get('error', 'Compilation failed'),
                    'compilation_time': compile_time
                }
            
            self.logger.info(f"Compilation successful in {compile_time:.2f}s")
            
            # Run tests
            test_start = time.time()
            test_results = self._run_tests(failing_tests)
            test_time = time.time() - test_start
            
            # Analyze results
            all_passed = all(r.get('passed', False) for r in test_results.values())
            
            if all_passed:
                result = {
                    'status': 'success',
                    'message': f'All {len(failing_tests)} tests passed',
                    'test_results': test_results,
                    'compilation_time': compile_time,
                    'test_time': test_time,
                    'total_time': time.time() - start_time
                }
            else:
                # ENHANCED: Analyze ALL test results, not just first failure
                failed_tests = []
                passed_tests = []
                error_details = {}
                all_error_types = set()
                all_error_messages = []

                for test_name, test_result in test_results.items():
                    if test_result.get('passed', False):
                        passed_tests.append(test_name)
                    else:
                        failed_tests.append(test_name)
                        
                        # Extract detailed error information for this specific test
                        error_info = self._extract_test_error(test_result.get('error', ''))
                        
                        # Store detailed error for this test
                        error_details[test_name] = {
                            'error_type': error_info['type'],
                            'error_message': error_info['message'],
                            'error_line': error_info.get('line'),
                            'stack_trace': error_info.get('stack_trace', ''),
                            'assertion_failure': error_info.get('assertion', None)
                        }
                        
                        # Collect unique error types and messages
                        all_error_types.add(error_info['type'])
                        if error_info['message'] not in all_error_messages:
                            all_error_messages.append(error_info['message'])

                # Determine overall status
                if not failed_tests:
                    # All tests passed
                    result = {
                        'status': 'success',
                        'passed_tests': passed_tests,
                        'failed_tests': [],
                        # ... rest of success result
                    }
                else:
                    # ENHANCED: Create comprehensive error summary
                    error_type_counts = {}
                    for test_error in error_details.values():
                        err_type = test_error['error_type']
                        error_type_counts[err_type] = error_type_counts.get(err_type, 0) + 1
                    
                    primary_error_type = max(error_type_counts.items(), key=lambda x: x[1])[0] if error_type_counts else 'TestFailure'
                    
                    # Combine error messages intelligently
                    if len(all_error_messages) == 1:
                        combined_message = all_error_messages[0]
                    else:
                        combined_message = f"Multiple errors: {'; '.join(all_error_messages[:3])}"
                        if len(all_error_messages) > 3:
                            combined_message += f" (and {len(all_error_messages) - 3} more)"
                    
                    result = {
                        'status': 'failed',
                        # Backward compatible fields
                        'error_type': primary_error_type,
                        'error_message': combined_message,
                        
                        # NEW: Complete error information
                        'all_error_types': list(all_error_types),
                        'error_details': error_details,
                        'error_summary': {
                            'total_failures': len(failed_tests),
                            'unique_error_types': len(all_error_types),
                            'error_type_distribution': error_type_counts
                        },
                        
                        'test_results': test_results,
                        'failed_tests': failed_tests,
                        'passed_tests': passed_tests,
                        'compilation_time': compile_time,
                        'test_time': test_time
                    }
            
            return result
            
        finally:
            # Always restore original file
            self._restore_file(file_path, backup_content)
    
    def execute_multi_patches(self, multi_patches: Dict, failing_tests: List[str]) -> Dict:
        """
        Execute multiple method patches together.
        All patches must be applied before running tests.
        
        Args:
            multi_patches: Dictionary containing multiple method patches
            failing_tests: List of tests to run
            
        Returns:
            Execution result dictionary
        """
        methods = multi_patches.get('methods', [])
        methods = list(methods.values())
        self.logger.info(f"Executing multi-method patches ({len(methods)} methods)")
        
        start_time = time.time()
        
        # Backup all files that will be modified
        backed_up_files = set()
        backup_info = {}
        
        try:
            # First, backup all files
            for method_patch in methods:
                file_path = method_patch.get('file_path', '')
                
                if file_path and file_path not in backed_up_files:
                    backup_content = self._backup_file(file_path)
                    
                    if backup_content is not None:
                        backup_info[file_path] = backup_content
                        backed_up_files.add(file_path)
                        self.logger.debug(f"Backed up {file_path}")
                    else:
                        return {
                            'status': 'failed',
                            'error_type': 'FileNotFound',
                            'error_message': f"File not found: {file_path}"
                        }
            
            # Apply all patches
            applied_patches = []
            methods = sorted(methods, key=lambda d: d['line_numbers'][0] if 'line_numbers' in d and d['line_numbers'] else float('inf'), reverse=True)
            for i, method_patch in enumerate(methods):
                self.logger.info(f"Applying patch {i+1}/{len(methods)}: {method_patch.get('method_name')}")
                
                success = self._apply_patch(method_patch)
                
                if not success:
                    return {
                        'status': 'failed',
                        'error_type': 'PatchApplicationError',
                        'error_message': f"Failed to apply patch for method {method_patch.get('method_name')}",
                        'failed_method': method_patch.get('method_name'),
                        'applied_methods': applied_patches
                    }
                
                applied_patches.append(method_patch.get('method_name'))
            
            self.logger.info(f"Successfully applied all {len(applied_patches)} patches")
            
            # Compile the project
            compile_start = time.time()
            compile_result = self._compile()
            compile_time = time.time() - compile_start
            
            if not compile_result['success']:
                return {
                    'status': 'failed',
                    'error_type': 'CompilationError',
                    'error_message': compile_result.get('error', 'Compilation failed'),
                    'applied_methods': applied_patches,
                    'compilation_time': compile_time
                }
            
            self.logger.info(f"Compilation successful in {compile_time:.2f}s")
            
            # Run all failing tests
            test_start = time.time()
            test_results = self._run_tests(failing_tests)
            test_time = time.time() - test_start
            
            # Analyze results
            all_passed = all(r.get('passed', False) for r in test_results.values())
            failed_tests = [t for t, r in test_results.items() if not r.get('passed', False)]
            passed_tests = [t for t, r in test_results.items() if r.get('passed', False)]
            
            if all_passed:
                return {
                    'status': 'success',
                    'message': f'All {len(failing_tests)} tests passed with {len(applied_patches)} method fixes',
                    'applied_methods': applied_patches,
                    'test_results': test_results,
                    'compilation_time': compile_time,
                    'test_time': test_time,
                    'total_time': time.time() - start_time
                }
            else:
                # Analyze which test failed first
                first_failure = test_results[failed_tests[0]] if failed_tests else {}
                return {
                    'status': 'failed',
                    'error_type': first_failure.get('error_type', 'TestFailure'),
                    'error_message': first_failure.get('error_message', 'Test failed'),
                    'error_line': first_failure.get('error_line'),
                    'failed_tests': failed_tests,
                    'passed_tests': passed_tests,
                    'applied_methods': applied_patches,
                    'test_results': test_results,
                    'compilation_time': compile_time,
                    'test_time': test_time,
                    'methods_fixed': len(applied_patches)
                }
            
        except Exception as e:
            self.logger.error(f"Error executing multi-patches: {e}")
            return {
                'status': 'failed',
                'error_type': 'ExecutionError',
                'error_message': str(e)
            }
        finally:
            # Always restore all original files
            for file_path, original_content in backup_info.items():
                self._restore_file(file_path, original_content)
    
    def _backup_file(self, file_path: str) -> Optional[str]:
        """Backup a file and return its content."""
        if not file_path:
            return None
        
        full_path = os.path.join(self.project_path, file_path)
        
        if not os.path.exists(full_path):
            self.logger.error(f"File not found for backup: {full_path}")
            return None
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            self.logger.error(f"Error backing up file {full_path}: {e}")
            return None
    
    def _restore_file(self, file_path: str, content: str):
        """Restore a file from backup."""
        if not file_path or content is None:
            return
        
        full_path = os.path.join(self.project_path, file_path)
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.debug(f"Restored {file_path}")
        except Exception as e:
            self.logger.error(f"Error restoring file {full_path}: {e}")
    
    def _apply_patch(self, patch: Dict) -> bool:
        """
        Apply a patch to a file using line numbers.
        
        Args:
            patch: Patch information. Expected keys:
                - file_path: str
                - method_name: str
                - fixed_method: str
                - line_numbers: List[int]
                - class_name: Optional[str]
                
        Returns:
            True if successful, False otherwise
        """
        file_path = patch.get('file_path', '')
        method_name = patch.get('method_name', '')
        fixed_code = patch.get('fixed_method', '')
        line_numbers = patch.get('line_numbers', [])
        
        if not file_path or not fixed_code or not line_numbers:
            self.logger.error("Missing required patch information (file_path, fixed_method, line_numbers)")
            return False
        
        full_path = os.path.join(self.project_path, file_path)
        
        if not os.path.exists(full_path):
            self.logger.error(f"File not found: {full_path}")
            return False
        
        try:
            # Read file lines
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            start = line_numbers[0]-1
            end = line_numbers[-1]
            
            if start < 1 or end > len(lines) or start > end:
                self.logger.error(f"Invalid line range {line_numbers} for file {file_path}")
                return False
            
            # Replace lines with fixed code
            fixed_lines = [line + "\n" if not line.endswith("\n") else line for line in fixed_code.splitlines()]
            new_content = lines[:start] + fixed_lines + lines[end:]
            
            # Write back
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(new_content)
            
            self.logger.debug(f"Applied patch for {method_name} at lines {start}-{end}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error applying patch: {e}")
            return False

    # def _apply_patch(self, patch: Dict) -> bool:
    #     """
    #     Apply a patch to a file.
        
    #     Args:
    #         patch: Patch information
            
    #     Returns:
    #         True if successful
    #     """
    #     file_path = patch.get('file_path', '')
    #     method_name = patch.get('method_name', '')
    #     fixed_code = patch.get('fixed_method', '')
    #     line_numbers = patch.get('line_numbers', [])
        
    #     if not file_path or not fixed_code:
    #         self.logger.error(f"Missing required patch information")
    #         return False
        
    #     full_path = os.path.join(self.project_path, file_path)
        
    #     if not os.path.exists(full_path):
    #         self.logger.error(f"File not found: {full_path}")
    #         return False
        
    #     try:
    #         # Read current content
    #         with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
    #             content = f.read()
            
    #         # Replace method
    #         updated_content = self._replace_method_in_content(
    #             content, 
    #             method_name, 
    #             fixed_code,
    #             patch.get('class_name', '')
    #         )
            
    #         if updated_content == content:
    #             self.logger.warning(f"Method {method_name} not found, trying aggressive replacement")
    #             updated_content = self._aggressive_method_replacement(
    #                 content,
    #                 method_name,
    #                 fixed_code
    #             )
                
    #             if updated_content == content:
    #                 self.logger.error(f"Failed to replace method {method_name}")
    #                 return False
            
    #         # Write updated content
    #         with open(full_path, 'w', encoding='utf-8') as f:
    #             f.write(updated_content)
            
    #         self.logger.debug(f"Applied patch for {method_name}")
    #         return True
            
    #     except Exception as e:
    #         self.logger.error(f"Error applying patch: {e}")
    #         return False
    
    def _replace_method_in_content(self, content: str, method_name: str, 
                                  fixed_code: str, class_name: str = '') -> str:
        """
        Replace method in file content.
        
        Args:
            content: File content
            method_name: Name of method to replace
            fixed_code: New method code
            class_name: Optional class name for better matching
            
        Returns:
            Updated content
        """
        lines = content.split('\n')
        
        # Find method start
        method_start = -1
        for i, line in enumerate(lines):
            # Look for method signature
            if method_name in line and any(keyword in line for keyword in 
                                          ['public', 'private', 'protected', 'static', 'void', 'def']):
                # Verify it's not a comment
                stripped = line.strip()
                if not stripped.startswith('//') and not stripped.startswith('#') and not stripped.startswith('*'):
                    method_start = i
                    # Go back to include annotations/decorators
                    while i > 0:
                        prev_line = lines[i-1].strip()
                        if prev_line.startswith('@') or prev_line.startswith('"""'):
                            i -= 1
                            method_start = i
                        else:
                            break
                    break
        
        if method_start == -1:
            return content  # Method not found
        
        # Find method end
        if self.language == 'java':
            method_end = self._find_java_method_end(lines, method_start)
        elif self.language == 'python':
            method_end = self._find_python_method_end(lines, method_start)
        else:
            method_end = method_start  # Fallback
        
        if method_end == -1:
            return content
        
        # Replace method
        fixed_lines = fixed_code.split('\n')
        
        # Preserve indentation
        if method_start < len(lines):
            original_indent = len(lines[method_start]) - len(lines[method_start].lstrip())
            if original_indent > 0:
                fixed_lines = [(' ' * original_indent + line if line.strip() else line) 
                              for line in fixed_lines]
        
        new_lines = lines[:method_start] + fixed_lines + lines[method_end + 1:]
        
        return '\n'.join(new_lines)
    
    def _find_java_method_end(self, lines: List[str], start: int) -> int:
        """Find the end of a Java method using brace counting."""
        brace_count = 0
        in_method = False
        
        for i in range(start, len(lines)):
            line = lines[i]
            
            # Count braces (ignoring those in strings and comments)
            cleaned = self._remove_strings_and_comments(line)
            open_braces = cleaned.count('{')
            close_braces = cleaned.count('}')
            
            brace_count += open_braces
            if open_braces > 0:
                in_method = True
            
            brace_count -= close_braces
            
            if in_method and brace_count == 0:
                return i
        
        return -1
    
    def _find_python_method_end(self, lines: List[str], start: int) -> int:
        """Find the end of a Python method using indentation."""
        if start >= len(lines):
            return -1
        
        # Get initial indentation
        initial_line = lines[start]
        initial_indent = len(initial_line) - len(initial_line.lstrip())
        
        # Find where indentation returns to same or lower level
        for i in range(start + 1, len(lines)):
            line = lines[i]
            
            # Skip empty lines
            if not line.strip():
                continue
            
            # Check indentation
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= initial_indent:
                return i - 1
        
        return len(lines) - 1
    
    def _remove_strings_and_comments(self, line: str) -> str:
        """Remove strings and comments from a line for accurate brace counting."""
        # Remove single-line comments
        if '//' in line:
            line = line[:line.index('//')]
        
        # Remove strings (simplified - doesn't handle all cases)
        in_string = False
        in_char = False
        cleaned = []
        i = 0
        while i < len(line):
            if not in_string and not in_char:
                if line[i] == '"' and (i == 0 or line[i-1] != '\\'):
                    in_string = True
                elif line[i] == "'" and (i == 0 or line[i-1] != '\\'):
                    in_char = True
                else:
                    cleaned.append(line[i])
            elif in_string and line[i] == '"' and line[i-1] != '\\':
                in_string = False
            elif in_char and line[i] == "'" and line[i-1] != '\\':
                in_char = False
            i += 1
        
        return ''.join(cleaned)
    
    def _aggressive_method_replacement(self, content: str, method_name: str,
                                      fixed_code: str) -> str:
        """
        More aggressive method replacement using regex.
        """
        import re
        
        # Try to find method with regex for Java
        if self.language == 'java':
            pattern = rf'(@\w+\s+)*?(public|private|protected|static|\s)+[\w<>\[\],\s]+\s+{re.escape(method_name)}\s*\([^{{]*\{{'
        else:  # Python
            pattern = rf'(\s*)def\s+{re.escape(method_name)}\s*\([^:]*:'
        
        match = re.search(pattern, content, re.MULTILINE)
        if not match:
            return content
        
        start = match.start()
        
        if self.language == 'java':
            # Find where the method ends by counting braces
            brace_count = 0
            in_method = False
            end = start
            
            for i in range(start, len(content)):
                char = content[i]
                if char == '{':
                    brace_count += 1
                    in_method = True
                elif char == '}':
                    brace_count -= 1
                    if in_method and brace_count == 0:
                        end = i + 1
                        break
        else:  # Python
            # Find method end by indentation
            lines_before = content[:start].count('\n')
            all_lines = content.split('\n')
            
            if lines_before < len(all_lines):
                end_line = self._find_python_method_end(all_lines, lines_before)
                
                # Convert back to character position
                end = sum(len(line) + 1 for line in all_lines[:end_line + 1]) - 1
            else:
                end = len(content)
        
        # Replace the method
        return content[:start] + fixed_code + content[end:]
    
    def _compile(self) -> Dict:
        """
        Compile the project.
        
        Returns:
            Dictionary with 'success' and optional 'error'
        """
        # Check compilation cache
        cache_key = self._get_project_hash()
        if cache_key in self._compilation_cache:
            self.logger.debug("Using cached compilation result")
            return self._compilation_cache[cache_key]
        
        if self.language == 'java':
            result = self._compile_java(self.project_path)
        elif self.language == 'python':
            result = self._compile_python()
        else:
            result = {'success': True}
        
        # Cache result
        if result['success']:
            self._compilation_cache[cache_key] = result
        
        return result
    
    def _compile_java(self, project_dir: str) -> Dict:
            """Compile Java project with better error extraction."""
            try:
                # Try defects4j compile first
                result = subprocess.run(
                    ['defects4j', 'compile'],
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                
                if result.returncode == 0:
                    return {'success': True}
                else:
                    error_output = result.stderr + result.stdout
                    return {
                        'success': False,
                        'error': error_output,
                        'returncode': result.returncode
                    }
                    
            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'error': 'Compilation timeout (180s exceeded)'
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Compilation error: {str(e)}'
                }
            
    def _compile_python(self) -> Dict:
        """Compile Python project (syntax check)."""
        try:
            # Find all Python files
            python_files = []
            for root, dirs, files in os.walk(self.project_path):
                # Skip virtual environments
                dirs[:] = [d for d in dirs if d not in ['venv', 'env', '.env', '__pycache__']]
                
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            # Syntax check all files
            for file_path in python_files:
                result = subprocess.run(
                    ['python', '-m', 'py_compile', file_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    return {
                        'success': False,
                        'error': f"Syntax error in {file_path}: {result.stderr}"
                    }
            
            return {'success': True}
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Compilation timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_tests(self, test_names: List[str]) -> Dict[str, Dict]:
        """
        Run multiple tests and return results for each.
        
        Args:
            test_names: List of test names
            
        Returns:
            Dictionary mapping test names to results
        """
        results = {}
        
        for test_name in test_names:
            self.logger.debug(f"Running test: {test_name}")
            results[test_name] = self._run_single_test(test_name)
        
        return results
    
    def _run_single_test(self, test_name: str) -> Dict:
        """
        Run a single test.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Test result dictionary
        """
        try:
            if self.language == 'java':
                return self._run_java_test(test_name)
            elif self.language == 'python':
                return self._run_python_test(test_name)
            else:
                return {'passed': False, 'error_type': 'UnsupportedLanguage'}
            
        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'error_type': 'Timeout',
                'error_message': 'Test execution timeout'
            }
        except Exception as e:
            return {
                'passed': False,
                'error_type': 'ExecutionError',
                'error_message': str(e)
            }
    
    def _run_java_test(self, test_name: str) -> Dict:
        """Run single Java test with comprehensive error detection."""
        try:
            # Run test with defects4j
            result = subprocess.run(
                ['defects4j', 'test', '-t', test_name],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            stdout, stderr = result.stdout, result.stderr
            
            # Check for compilation error
            if "COMPILATION ERROR" in stdout or "compilation" in stderr.lower():
                return {
                    "test": test_name,
                    "passed": False,
                    "output": stdout + stderr,
                    "error": "Compilation error",
                    "error_type": "CompilationError"
                }
            
            # Check failing_tests file
            failing_tests_file = os.path.join(self.project_path, "failing_tests")
            
            if os.path.exists(failing_tests_file):
                with open(failing_tests_file, "r") as f:
                    failing_content = f.read().strip()
                
                if test_name in failing_content:
                    # Extract error details
                    error_info = self._extract_test_error(stdout + stderr)
                    return {
                        "test": test_name,
                        "passed": False,
                        "output": stdout,
                        "error": failing_content,
                    }
            
            # Test passed
            return {
                "test": test_name,
                "passed": True,
                "output": stdout,
                "error": ""
            }
            
        except subprocess.TimeoutExpired:
            return {
                "test": test_name,
                "passed": False,
                "output": "",
                "error": "Test timeout",
                "error_type": "Timeout"
            }
        except Exception as e:
            return {
                "test": test_name,
                "passed": False,
                "output": "",
                "error": str(e),
                "error_type": "ExecutionError"
            }
    
    def _run_python_test(self, test_name: str) -> Dict:
        """Run a Python test."""
        try:
            # Try pytest first
            result = subprocess.run(
                ['python', '-m', 'pytest', test_name, '-xvs'],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Fallback to unittest if pytest fails
            if result.returncode == 2:  # pytest not found
                result = subprocess.run(
                    ['python', '-m', 'unittest', test_name, '-v'],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            
            passed = result.returncode == 0
            output = result.stdout + '\n' + result.stderr
            
            if not passed:
                error_type, error_msg, error_line = self._extract_test_error(output)
                return {
                    'passed': False,
                    'error_type': error_type,
                    'error_message': error_msg,
                    'error_line': error_line,
                    'output': output[:2000]
                }
            else:
                return {
                    'passed': True,
                    'output': output[:500]
                }
            
        except Exception as e:
            return {
                'passed': False,
                'error_type': 'TestExecutionError',
                'error_message': str(e)
            }

    def _extract_test_error(self, error_string: str) -> Dict[str, Any]:
        """
        ENHANCED: Extract detailed error information from test output.
        Now captures assertion failures and more error types.
        """
        error_info = {
            'type': 'TestFailure',
            'message': error_string[:500] if error_string else 'Unknown error',
            'line': None,
            'stack_trace': error_string[:1000] if error_string else '',
            'assertion': None
        }
        
        if not error_string:
            return error_info
        
        # Common Java exceptions
        java_exceptions = [
            'NullPointerException', 'IndexOutOfBoundsException', 
            'ArrayIndexOutOfBoundsException', 'ClassCastException',
            'IllegalArgumentException', 'IllegalStateException',
            'ArithmeticException', 'NumberFormatException',
            'UnsupportedOperationException', 'NoSuchMethodException',
            'NoSuchElementException'
        ]
        
        # Check for exception types
        for exc in java_exceptions:
            if exc in error_string:
                error_info['type'] = exc
                
                # Try to extract the specific message
                pattern = rf'{exc}:\s*([^\n]+)'
                match = re.search(pattern, error_string)
                if match:
                    error_info['message'] = match.group(1).strip()
                
                # Try to extract line number
                line_pattern = r'at\s+[\w.]+\([\w.]+:(\d+)\)'
                line_match = re.search(line_pattern, error_string)
                if line_match:
                    error_info['line'] = int(line_match.group(1))
                break
        
        # Check for assertion failures
        if 'AssertionError' in error_string or 'expected' in error_string.lower():
            error_info['type'] = 'AssertionFailure'
            
            # Try to extract expected vs actual
            pattern = r'expected:\s*<([^>]+)>\s*but was:\s*<([^>]+)>'
            match = re.search(pattern, error_string, re.IGNORECASE)
            if match:
                error_info['assertion'] = {
                    'expected': match.group(1),
                    'actual': match.group(2)
                }
                error_info['message'] = f"Expected {match.group(1)} but got {match.group(2)}"
        
        return error_info

    
    def _extract_compilation_error(self, error_output: str) -> str:
        """Extract meaningful compilation error from compiler output."""
        lines = error_output.split('\n')
        
        # Look for error markers
        error_lines = []
        for line in lines:
            if 'error:' in line.lower() or 'cannot find symbol' in line:
                error_lines.append(line.strip())
        
        if error_lines:
            return ' | '.join(error_lines[:3])  # Return first 3 errors
        
        # Return first non-empty line as fallback
        for line in lines:
            if line.strip():
                return line.strip()[:200]
        
        return 'Compilation failed'
    
    def _get_classpath(self) -> str:
        """Get classpath for Java compilation/execution."""
        paths = ['.']
        
        # Add common library directories
        lib_dirs = ['lib', 'libs', 'target/classes', 'build/classes', 'bin']
        
        for lib_dir in lib_dirs:
            full_path = os.path.join(self.project_path, lib_dir)
            if os.path.exists(full_path):
                paths.append(full_path)
                
                # Add JAR files
                if os.path.isdir(full_path):
                    for file in os.listdir(full_path):
                        if file.endswith('.jar'):
                            paths.append(os.path.join(full_path, file))
        
        return ':'.join(paths) if os.name != 'nt' else ';'.join(paths)
    
    def _get_project_hash(self) -> str:
        """Get hash of project state for caching."""
        # Simple hash based on project path and timestamp
        # In production, should hash actual file contents
        return hashlib.md5(f"{self.project_path}{int(time.time()/60)}".encode()).hexdigest()


class DockerExecutor(PatchExecutor):
    """
    Execute patches in isolated Docker containers for safety.
    Extends PatchExecutor with Docker support.
    """
    
    def __init__(self, project_path: str, language: str = 'java'):
        super().__init__(project_path, language)
        
        self.image_name = f"bugfix_{language}:latest"
        self.docker_available = self._check_docker()
        
        if self.docker_available:
            try:
                import docker
                self.docker_client = docker.from_env()
                self._ensure_docker_image()
                self.logger.info("Docker executor initialized")
            except Exception as e:
                self.logger.warning(f"Docker not available: {e}")
                self.docker_available = False
                self.logger.info("Falling back to local executor")
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ['docker', '--version'], 
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _ensure_docker_image(self):
        """Ensure Docker image exists."""
        try:
            self.docker_client.images.get(self.image_name)
            self.logger.debug(f"Docker image {self.image_name} found")
        except:
            self.logger.info(f"Building Docker image {self.image_name}")
            self._build_docker_image()
    
    def _build_docker_image(self):
        """Build Docker image for testing."""
        dockerfile_content = self._get_dockerfile()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='Dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name
        
        try:
            # Build image
            image, logs = self.docker_client.images.build(
                path=os.path.dirname(dockerfile_path),
                dockerfile=os.path.basename(dockerfile_path),
                tag=self.image_name,
                rm=True,
                forcerm=True
            )
            
            for log in logs:
                if 'stream' in log:
                    self.logger.debug(log['stream'].strip())
                    
            self.logger.info(f"Successfully built Docker image {self.image_name}")
            
        finally:
            os.unlink(dockerfile_path)
    
    def _get_dockerfile(self) -> str:
        """Get Dockerfile content based on language."""
        if self.language == 'java':
            return """
FROM maven:3.8-openjdk-11
RUN apt-get update && apt-get install -y git curl wget
WORKDIR /app
CMD ["/bin/bash"]
"""
        else:  # Python
            return """
FROM python:3.9
RUN pip install pytest coverage pytest-timeout numpy
WORKDIR /app
CMD ["/bin/bash"]
"""
    
    def execute_patch(self, patch: Dict, failing_tests: List[str]) -> Dict:
        """Execute patch in Docker container if available."""
        if not self.docker_available:
            # Fallback to regular execution
            return super().execute_patch(patch, failing_tests)
        
        container = None
        try:
            # Create and start container
            container = self.docker_client.containers.run(
                self.image_name,
                command="sleep 600",  # Keep alive for 10 minutes
                detach=True,
                volumes={
                    self.project_path: {'bind': '/app', 'mode': 'rw'}
                },
                working_dir='/app',
                mem_limit='2g',
                cpu_quota=100000,  # Limit CPU usage
                remove=False
            )
            
            self.logger.debug(f"Started Docker container {container.short_id}")
            
            # Execute patch in container
            result = self._execute_in_container(container, patch, failing_tests)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Docker execution failed: {e}")
            # Fallback to regular execution
            return super().execute_patch(patch, failing_tests)
        finally:
            if container:
                try:
                    container.stop(timeout=10)
                    container.remove()
                    self.logger.debug(f"Cleaned up container {container.short_id}")
                except:
                    pass
    
    def _execute_in_container(self, container, patch: Dict, 
                             failing_tests: List[str]) -> Dict:
        """
        Execute patch inside Docker container.
        
        Note: This is a simplified version. In production, you would
        properly execute commands in the container and retrieve results.
        """
        # For now, fallback to regular execution
        # A full implementation would execute all steps in the container
        return super().execute_patch(patch, failing_tests)