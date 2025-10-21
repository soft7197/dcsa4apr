"""
Agent-Based Automated Program Repair System for SWE-bench Lite
Complete implementation with multi-patch generation and immediate evaluation
"""

from datetime import datetime
import os
import json
import time
import subprocess
import hashlib
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import openai
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """System configuration"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_context_updater: str = "gpt-4o"
    model_generator: str = "gpt-4o"
    model_overfitting: str = "gpt-4o"
    temp_context_updater: float = 0.2
    temp_generator: float = 0.7
    temp_overfitting: float = 0.3
    max_tokens: int = 5000
    max_iterations: int = 5
    timeout_per_bug: int = 3600
    workspace_dir: str = "./workspace"
    results_dir: str = "./results"
    predictions_dir: str = "./predictions"
    use_swebench_eval: bool = True

config = Config()
openai.api_key = config.openai_api_key

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BugInstance:
    """Represents a bug from SWE-bench"""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    test_patch: str
    patch: str
    version: str
    fail_to_pass: List[str]
    pass_to_pass: List[str]
    environment_setup_commit: str = ""
    
@dataclass
class ContextItem:
    """Item in dynamic context"""
    content: str
    source: str
    timestamp: float
    iteration: int
    relevance_score: float = 1.0

@dataclass
class Hypothesis:
    """Generated patch hypothesis"""
    summary: str
    diff: str
    fixed_code: Dict[str, str]
    test_results: Dict = field(default_factory=dict)
    semantic_hash: str = ""
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)
    model_patch: str = ""

@dataclass
class IterationLog:
    """Log for each iteration"""
    iteration: int
    timestamp: float
    prompt: str
    hypotheses: List[Dict]
    error_messages: List[str]
    success: bool
    successful_hypothesis: Optional[Dict] = None
    tool_executions: List[Dict] = field(default_factory=list)  # ADD THIS LINE


@dataclass
class ContextPool:
    """Central context repository"""
    # Static
    original_code: Dict[str, str] = field(default_factory=dict)
    buggy_methods: Dict[str, str] = field(default_factory=dict)
    problem_statement: str = ""
    failing_tests: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    
    # Dynamic
    dynamic_context: List[ContextItem] = field(default_factory=list)
    
    # History
    tried_hypotheses: List[Hypothesis] = field(default_factory=list)
    tool_extractions: Dict[str, List] = field(default_factory=dict)
    iteration_logs: List[IterationLog] = field(default_factory=list)

# ============================================================================
# Buggy Method Extractor
# ============================================================================

class BuggyMethodExtractor:
    """Extract buggy methods from patch"""
    
    @staticmethod
    def extract_from_patch(patch: str, repo_path: Path) -> Dict[str, str]:
        """Extract buggy methods mentioned in the patch"""
        buggy_methods = {}
        
        # Parse patch to find modified files and line ranges
        file_pattern = r'---\s+a/(.*?)\s+\+\+\+\s+b/(.*?)(?:\n|$)'
        matches = re.finditer(file_pattern, patch, re.MULTILINE)
        
        for match in matches:
            file_path = match.group(1)
            full_path = repo_path / file_path
            
            if not full_path.exists() or not file_path.endswith('.py'):
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
                
                # Get modified line numbers from this file's hunks
                modified_lines = BuggyMethodExtractor._get_modified_lines(patch, file_path)
                modified_lines = set(list(modified_lines)[:-1])
                if not modified_lines:
                    continue
                
                # Parse the file and find which methods contain these lines
                modified_methods = BuggyMethodExtractor._find_methods_containing_lines(
                    file_content, modified_lines, file_path
                )
                
                for method_name, method_code in modified_methods.items():
                    key = f"{file_path}::{method_name}"
                    buggy_methods[key] = method_code
                    
            except Exception as e:
                print(f"  Warning: Could not extract methods from {file_path}: {e}")
        
        return buggy_methods

    @staticmethod
    def _get_modified_lines(patch: str, file_path: str) -> set:
        """Extract line numbers that were modified in the patch for a specific file"""
        modified_lines = set()
        
        # Find the section of the patch for this file
        file_section_pattern = rf'---\s+a/{re.escape(file_path)}.*?\n\+\+\+.*?\n(.*?)(?=\n---|\n\+\+\+|\Z)'
        file_match = re.search(file_section_pattern, patch, re.DOTALL)
        
        if not file_match:
            return modified_lines
        
        file_patch = file_match.group(1)
        
        # Parse hunks to get modified line ranges
        hunk_pattern = r'@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@'
        
        for match in re.finditer(hunk_pattern, file_patch):
            start_line = int(match.group(1))
            line_count = int(match.group(2)) if match.group(2) else 1
            
            # Add all lines in this range
            for line_num in range(start_line, start_line + line_count):
                modified_lines.add(line_num)
        
        return modified_lines
            
    @staticmethod
    def _find_methods_containing_lines(file_content: str, modified_lines: set, file_path: str) -> Dict[str, str]:
        """Find which methods contain the modified lines"""
        methods = {}
        
        try:
            tree = ast.parse(file_content)
            lines = file_content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = node.name
                    method_start = node.lineno
                    method_end = node.end_lineno if hasattr(node, 'end_lineno') else method_start + 50
                    
                    # Check if any modified line falls within this method
                    method_lines = set(range(method_start, method_end + 1))
                    if modified_lines & method_lines:  # Intersection
                        # This method contains modified lines
                        method_code = '\n'.join(lines[method_start - 1:method_end])
                        methods[method_name] = method_code
                        
        except SyntaxError:
            # Fallback to regex if AST parsing fails
            methods = BuggyMethodExtractor._find_methods_regex_with_lines(
                file_content, modified_lines
            )
        
        return methods
    
    @staticmethod
    def _find_methods_regex_with_lines(file_content: str, modified_lines: set) -> Dict[str, str]:
        """Fallback method extraction using regex"""
        methods = {}
        lines = file_content.split('\n')
        
        # Find all function definitions with their line numbers
        func_pattern = r'^( *)(?:async\s+)?def\s+(\w+)\s*\('
        
        for line_num, line in enumerate(lines, 1):
            match = re.match(func_pattern, line)
            if match:
                indent = len(match.group(1))
                method_name = match.group(2)
                
                # Find the end of this method
                method_end = len(lines)
                for end_line_num in range(line_num + 1, len(lines) + 1):
                    if end_line_num <= len(lines):
                        next_line = lines[end_line_num - 1]
                        if next_line.strip() and not next_line.startswith(' ' * (indent + 1)):
                            if re.match(r'^\s*def\s+', next_line) or re.match(r'^\s*class\s+', next_line):
                                method_end = end_line_num - 1
                                break
                
                # Check if any modified line is in this method
                method_range = set(range(line_num, method_end + 1))
                if modified_lines & method_range:
                    method_code = '\n'.join(lines[line_num - 1:method_end])
                    methods[method_name] = method_code[:1000]  # Limit size
        
        return methods
    
    @staticmethod
    def _extract_methods_from_file(content: str, file_path: str) -> Dict[str, str]:
        """Extract all methods from a Python file"""
        methods = {}
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = node.name
                    
                    # Get the method source code
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                    
                    lines = content.split('\n')
                    method_code = '\n'.join(lines[start_line:end_line])
                    
                    methods[method_name] = method_code
                    
        except SyntaxError:
            # If AST parsing fails, use regex as fallback
            methods = BuggyMethodExtractor._extract_methods_regex(content)
        
        return methods
    
    @staticmethod
    def _extract_methods_regex(content: str) -> Dict[str, str]:
        """Fallback method extraction using regex"""
        methods = {}
        
        # Match function definitions
        pattern = r'^( *)(?:async\s+)?def\s+(\w+)\s*\([^)]*\).*?(?=\n\1(?:def|class|$)|\Z)'
        
        for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
            method_name = match.group(2)
            method_code = match.group(0)
            methods[method_name] = method_code[:500]  # Limit size
        
        return methods

# ============================================================================
# Tools Suite
# ============================================================================

class ToolCommandValidator:
    """Validates and corrects tool commands from LLM"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self._build_file_index()
    
    def _build_file_index(self):
        """Build index of all Python files and their methods"""
        self.file_index = {}
        self.method_index = {}
        
        try:
            for py_file in self.repo_path.rglob("*.py"):
                rel_path = str(py_file.relative_to(self.repo_path))
                self.file_index[rel_path] = py_file
                
                # Also store just filename for partial matching
                self.file_index[py_file.name] = py_file
                
                # Extract method names
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        tree = ast.parse(content)
                        
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                method_key = f"{rel_path}::{node.name}"
                                self.method_index[node.name] = rel_path
                                self.method_index[method_key] = rel_path
                except:
                    pass
        except Exception as e:
            print(f"  Warning: Error building file index: {e}")
    
    def validate_and_fix_command(self, tool_name: str, params: Dict) -> Tuple[bool, Dict, str]:
        """
        Validate and fix tool command parameters
        Returns: (is_valid, fixed_params, error_message)
        """
        if tool_name == "run_coverage":
            return self._validate_coverage_params(params)
        elif tool_name == "search_similar_methods":
            return self._validate_similarity_params(params)
        elif tool_name == "extract_code":
            return self._validate_extract_params(params)
        elif tool_name == "get_call_graph":
            return self._validate_call_graph_params(params)
        elif tool_name == "analyze_dependencies":
            return self._validate_dependencies_params(params)
        elif tool_name == "find_api_usage":
            return self._validate_api_usage_params(params)
        else:
            return False, params, f"Unknown tool: {tool_name}"
    
    def _validate_coverage_params(self, params: Dict) -> Tuple[bool, Dict, str]:
        """Validate coverage command parameters"""
        test_name = params.get("test_name", "")
        
        if not test_name:
            return False, params, "test_name is required"
        
        # Test name should be in format: path/to/test_file.py::test_function
        if "::" in test_name:
            file_part, test_part = test_name.split("::", 1)
            fixed_file = self._resolve_file_path(file_part)
            if fixed_file:
                params["test_name"] = f"{fixed_file}::{test_part}"
        else:
            # Just a file path
            fixed_file = self._resolve_file_path(test_name)
            if fixed_file:
                params["test_name"] = fixed_file
        
        return True, params, ""
    
    def _validate_similarity_params(self, params: Dict) -> Tuple[bool, Dict, str]:
        """Validate similarity search parameters"""
        code = params.get("code", "")
        k = params.get("k", 5)
        
        if not code or not isinstance(code, str):
            return False, params, "code parameter must be non-empty string"
        
        # Ensure k is valid integer
        try:
            k = int(k)
            if k < 1:
                k = 5
            if k > 20:
                k = 20
            params["k"] = k
        except:
            params["k"] = 5
        
        return True, params, ""
    
    def _validate_extract_params(self, params: Dict) -> Tuple[bool, Dict, str]:
        """Validate code extraction parameters"""
        file_path = params.get("file_path", "")
        element_name = params.get("element_name")
        
        if not file_path:
            return False, params, "file_path is required"
        
        # Try to resolve file path
        fixed_path = self._resolve_file_path(file_path)
        if not fixed_path:
            return False, params, f"Could not resolve file path: {file_path}"
        
        params["file_path"] = fixed_path
        
        # If element_name is provided, validate it's a string
        if element_name is not None:
            if not isinstance(element_name, str) or not element_name.strip():
                return False, params, "element_name must be a non-empty string"
            params["element_name"] = element_name.strip()
        
        # Remove start_line and end_line if element_name is provided
        if element_name:
            params.pop("start_line", None)
            params.pop("end_line", None)
        else:
            # Validate line numbers if provided and no element_name
            start_line = params.get("start_line")
            end_line = params.get("end_line")
            
            if start_line is not None:
                try:
                    start_line = int(start_line)
                    if start_line < 1:
                        start_line = 1
                    params["start_line"] = start_line
                except:
                    params.pop("start_line", None)
            
            if end_line is not None:
                try:
                    end_line = int(end_line)
                    if end_line < 1:
                        end_line = None
                    params["end_line"] = end_line
                except:
                    params.pop("end_line", None)
        
        return True, params, ""
    
    def _validate_call_graph_params(self, params: Dict) -> Tuple[bool, Dict, str]:
        """Validate call graph parameters"""
        file_path = params.get("file_path", "")
        if 'function' in params:
            function_name = params.get("function", "") 
            params.pop("function", None)  # Remove old key
            params["function_name"] = function_name  # Use new key
        elif 'function_name' in params:
            function_name = params.get("function_name", "")
 
        if not file_path or not function_name:
            return False, params, "file_path and function_name are required"
        
        fixed_path = self._resolve_file_path(file_path)
        if not fixed_path:
            return False, params, f"Could not resolve file path: {file_path}"
        
        params["file_path"] = fixed_path
        return True, params, ""
    
    def _validate_dependencies_params(self, params: Dict) -> Tuple[bool, Dict, str]:
        """Validate dependency analysis parameters"""
        return self._validate_call_graph_params(params)  # Same validation
    
    def _validate_api_usage_params(self, params: Dict) -> Tuple[bool, Dict, str]:
        """Validate API usage search parameters"""
        api_name = params.get("api_name", "")
        
        if not api_name or not isinstance(api_name, str):
            return False, params, "api_name must be non-empty string"
        
        return True, params, ""
    
    def _resolve_file_path(self, file_path: str) -> Optional[str]:
        """Resolve partial or incorrect file paths"""
        if not file_path:
            return None
        
        # Clean the path
        file_path = file_path.strip().strip("'\"")
        
        # Try exact match first
        if file_path in self.file_index:
            resolved = self.file_index[file_path]
            return str(resolved.relative_to(self.repo_path))
        
        # Try as relative path from repo root
        full_path = self.repo_path / file_path
        if full_path.exists():
            return file_path
        
        # Try finding by filename
        filename = Path(file_path).name
        if filename in self.file_index:
            resolved = self.file_index[filename]
            return str(resolved.relative_to(self.repo_path))
        
        # Try partial matching
        file_path_lower = file_path.lower()
        for indexed_path, actual_path in self.file_index.items():
            if isinstance(indexed_path, str):
                if file_path_lower in indexed_path.lower():
                    return str(actual_path.relative_to(self.repo_path))
        
        return None


class ToolSuite:
    """Integrated tool suite for repair context"""
    
    def __init__(self, repo_path: str, instance_id: str):
        self.repo_path = Path(repo_path)
        self.instance_id = instance_id
        self.validator = ToolCommandValidator(self.repo_path)
        self._build_method_index()
        
    def _build_method_index(self):
        """Build index of all methods for fast text similarity"""
        self.methods_data = []
        
        try:
            for py_file in self.repo_path.rglob("*.py"):
                rel_path = str(py_file.relative_to(self.repo_path))
                
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            start_line = node.lineno - 1
                            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
                            
                            method_code = '\n'.join(lines[start_line:end_line])
                            
                            self.methods_data.append({
                                'file': rel_path,
                                'name': node.name,
                                'code': method_code,
                                'start_line': node.lineno,
                                'end_line': end_line + 1
                            })
                except:
                    continue
        except Exception as e:
            print(f"  Warning: Error building method index: {e}")
    
    def execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """Execute tool with validation"""
        # Validate and fix parameters
        is_valid, fixed_params, error_msg = self.validator.validate_and_fix_command(
            tool_name, params
        )
        
        if not is_valid:
            return {
                "success": False,
                "error": f"Invalid parameters: {error_msg}",
                "tool": tool_name
            }
        
        # Execute the tool
        try:
            if tool_name == "run_coverage":
                return self.run_coverage(**fixed_params)
            elif tool_name == "search_similar_methods":
                return self.search_similar_methods(**fixed_params)
            elif tool_name == "extract_code":
                return self.extract_code(**fixed_params)
            elif tool_name == "get_call_graph":
                return self.get_call_graph(**fixed_params)
            elif tool_name == "analyze_dependencies":
                return self.analyze_dependencies(**fixed_params)
            elif tool_name == "find_api_usage":
                return self.find_api_usage(**fixed_params)
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool": tool_name
            }
    
    def run_coverage(self, test_name: str) -> Dict:
        """Run test with coverage analysis using SWE-bench harness"""
        try:
            # Generate unique run_id
            run_id = f"coverage_{self.instance_id}_{int(time.time())}"
            
            # Create minimal predictions file
            pred_file = Path(f"/tmp/coverage_pred_{run_id}.jsonl")
            prediction = {
                "instance_id": self.instance_id,
                "model_patch": "",  # Empty patch, just running test
                "model_name_or_path": "coverage_tool"
            }
            
            with open(pred_file, 'w') as f:
                json.dump(prediction, f)
                f.write('\n')
            
            # Run coverage using SWE-bench harness
            cmd = [
                "python", "-m", "swebench.harness.run_evaluation",
                "--dataset_name", "princeton-nlp/SWE-bench_Lite",
                "--instance_ids", self.instance_id,
                "--custom_test_command", f"coverage run --append -m pytest {test_name}",
                "--max_workers", "1",
                "--run_id", run_id,
                "--predictions_path", str(pred_file),
                "--timeout", "300"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=400
            )
            
            # Parse coverage file
            coverage_dir = Path(f"/home/selab/Desktop/swe_exp/logs/run_evaluation/{run_id}/coverage_tool/{self.instance_id}")
            coverage_file = coverage_dir / ".coverage"
            
            if not coverage_file.exists():
                return {
                    "success": False,
                    "error": "Coverage file not generated",
                    "stdout": result.stdout[-500:],
                    "stderr": result.stderr[-500:]
                }
            
            # Parse coverage data
            executed_lines = self._parse_coverage_file(coverage_file)
            
            # Clean up
            pred_file.unlink(missing_ok=True)
            
            return {
                "success": True,
                "test_name": test_name,
                "executed_lines": executed_lines,
                "coverage_file": str(coverage_file)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Coverage analysis timed out"
            }
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": f"Coverage failed: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def _parse_coverage_file(self, coverage_file: Path) -> Dict[str, List[int]]:
        """Parse .coverage file and extract executed lines"""
        try:
            import coverage
            
            cov = coverage.Coverage(data_file=str(coverage_file))
            cov.load()
            
            executed_lines = {}
            
            # Get all measured files
            for filename in cov.get_data().measured_files():
                # Convert to relative path
                try:
                    rel_path = str(Path(filename).relative_to(self.repo_path))
                except ValueError:
                    continue
                
                # Get executed line numbers
                analysis = cov.analysis2(filename)
                executed = analysis[1]  # Executed lines
                
                if executed:
                    executed_lines[rel_path] = sorted(list(executed))
            
            return executed_lines
            
        except Exception as e:
            print(f"  Warning: Could not parse coverage file: {e}")
            return {}
    
    def search_similar_methods(self, code: str, k: int = 5) -> Dict:
        """Find similar methods using fast text similarity (TF-IDF)"""
        if not self.methods_data:
            return {
                "success": False,
                "error": "No methods indexed",
                "results": []
            }
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Prepare corpus
            corpus = [m['code'] for m in self.methods_data]
            corpus.append(code)  # Add query
            
            # Compute TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Compute similarity with query (last item)
            query_vector = tfidf_matrix[-1]
            similarities = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()
            
            # Get top k
            top_indices = similarities.argsort()[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold
                    method = self.methods_data[idx].copy()
                    method['similarity'] = float(similarities[idx])
                    results.append(method)
            
            return {
                "success": True,
                "query": code[:100],
                "num_results": len(results),
                "results": results[1:]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Similarity search failed: {str(e)}",
                "results": []
            }
    
    def extract_code(self, file_path: str, element_name: str = None, start_line: int = None, end_line: int = None) -> Dict:
        """
        Extract code from file by element name or line range
        
        Args:
            file_path: Path to the Python file
            element_name: Name of method, field, or constructor to extract
            start_line: Start line for manual extraction (if element_name not provided)
            end_line: End line for manual extraction (if element_name not provided)
        
        Returns:
            Dict with success status and extracted code
        """
        try:
            full_path = self.repo_path / file_path
            
            if not full_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # If element_name is provided, extract that specific element
            if element_name:
                return self._extract_element_by_name(
                    content, 
                    lines, 
                    element_name, 
                    file_path
                )
            
            # Otherwise, use line range or return full file
            if start_line is not None and end_line is not None:
                extracted = '\n'.join(lines[start_line-1:end_line])
                return {
                    "success": True,
                    "file": file_path,
                    "element_type": "line_range",
                    "start_line": start_line,
                    "end_line": end_line,
                    "code": extracted,
                    "total_lines": len(lines)
                }
            else:
                return {
                    "success": True,
                    "file": file_path,
                    "element_type": "full_file",
                    "code": content,
                    "total_lines": len(lines)
                }
                
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": f"Failed to read file: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def _extract_element_by_name(self, content: str, lines: List[str], element_name: str, file_path: str) -> Dict:
        """Extract a specific element (method, field, or constructor) by name"""
        try:
            tree = ast.parse(content)
            
            # Search for the element
            for node in ast.walk(tree):
                # Check for methods (functions)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == element_name:
                        start_line = node.lineno
                        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 50
                        
                        code = '\n'.join(lines[start_line - 1:end_line])
                        
                        # Determine if it's a test method
                        is_test = element_name.startswith('test_') or any(
                            isinstance(dec, ast.Name) and dec.id in ['pytest', 'unittest']
                            for dec in node.decorator_list
                        )
                        
                        # Check if it's a constructor
                        is_constructor = element_name == '__init__'
                        
                        return {
                            "success": True,
                            "file": file_path,
                            "element_name": element_name,
                            "element_type": "constructor" if is_constructor else ("test_method" if is_test else "method"),
                            "start_line": start_line,
                            "end_line": end_line,
                            "code": code,
                            "signature": self._get_function_signature(node, lines),
                            "decorators": [self._get_decorator_name(dec) for dec in node.decorator_list],
                            "total_lines": len(lines)
                        }
                
                # Check for class definitions (for constructors or class-level fields)
                elif isinstance(node, ast.ClassDef):
                    # Check for __init__ inside this class
                    for class_node in node.body:
                        if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if class_node.name == element_name:
                                start_line = class_node.lineno
                                end_line = class_node.end_lineno if hasattr(class_node, 'end_lineno') else start_line + 50
                                
                                code = '\n'.join(lines[start_line - 1:end_line])
                                
                                is_constructor = element_name == '__init__'
                                
                                return {
                                    "success": True,
                                    "file": file_path,
                                    "element_name": element_name,
                                    "element_type": "constructor" if is_constructor else "method",
                                    "class_name": node.name,
                                    "start_line": start_line,
                                    "end_line": end_line,
                                    "code": code,
                                    "signature": self._get_function_signature(class_node, lines),
                                    "decorators": [self._get_decorator_name(dec) for dec in class_node.decorator_list],
                                    "total_lines": len(lines)
                                }
                        
                        # Check for field assignments (class variables)
                        elif isinstance(class_node, ast.Assign):
                            for target in class_node.targets:
                                if isinstance(target, ast.Name) and target.id == element_name:
                                    line_num = class_node.lineno
                                    code = lines[line_num - 1].strip()
                                    
                                    return {
                                        "success": True,
                                        "file": file_path,
                                        "element_name": element_name,
                                        "element_type": "field",
                                        "class_name": node.name,
                                        "line_number": line_num,
                                        "code": code,
                                        "total_lines": len(lines)
                                    }
                        
                        # Check for annotated assignments (typed fields)
                        elif isinstance(class_node, ast.AnnAssign):
                            if isinstance(class_node.target, ast.Name) and class_node.target.id == element_name:
                                line_num = class_node.lineno
                                code = lines[line_num - 1].strip()
                                
                                return {
                                    "success": True,
                                    "file": file_path,
                                    "element_name": element_name,
                                    "element_type": "field",
                                    "class_name": node.name,
                                    "line_number": line_num,
                                    "code": code,
                                    "annotation": ast.unparse(class_node.annotation) if hasattr(ast, 'unparse') else str(class_node.annotation),
                                    "total_lines": len(lines)
                                }
            
            # If not found in AST, try module-level search
            for node in tree.body:
                # Module-level function
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == element_name:
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 50
                    code = '\n'.join(lines[start_line - 1:end_line])
                    
                    return {
                        "success": True,
                        "file": file_path,
                        "element_name": element_name,
                        "element_type": "function",
                        "start_line": start_line,
                        "end_line": end_line,
                        "code": code,
                        "signature": self._get_function_signature(node, lines),
                        "total_lines": len(lines)
                    }
                
                # Module-level variable
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == element_name:
                            line_num = node.lineno
                            code = lines[line_num - 1].strip()
                            
                            return {
                                "success": True,
                                "file": file_path,
                                "element_name": element_name,
                                "element_type": "variable",
                                "line_number": line_num,
                                "code": code,
                                "total_lines": len(lines)
                            }
            
            # Element not found
            return {
                "success": False,
                "error": f"Element '{element_name}' not found in {file_path}",
                "file": file_path,
                "element_name": element_name
            }
            
        except SyntaxError as e:
            # Fallback to regex-based extraction
            return self._extract_element_by_name_regex(lines, element_name, file_path)
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": f"Failed to extract element: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def _get_function_signature(self, node: ast.FunctionDef, lines: List[str]) -> str:
        """Extract function signature from AST node"""
        try:
            if hasattr(ast, 'unparse'):
                # Python 3.9+
                args_str = ast.unparse(node.args)
                return f"def {node.name}({args_str})"
            else:
                # Fallback: get from source
                line = lines[node.lineno - 1].strip()
                return line.rstrip(':')
        except:
            return f"def {node.name}(...)"
    
    def _get_decorator_name(self, decorator) -> str:
        """Extract decorator name from AST node"""
        try:
            if isinstance(decorator, ast.Name):
                return decorator.id
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    return decorator.func.id
                elif isinstance(decorator.func, ast.Attribute):
                    return decorator.func.attr
            elif isinstance(decorator, ast.Attribute):
                return decorator.attr
            return str(decorator)
        except:
            return "unknown"
    
    def _extract_element_by_name_regex(self, lines: List[str], element_name: str, file_path: str) -> Dict:
        """Fallback: Extract element using regex when AST parsing fails"""
        
        # Try to find function/method definition
        func_pattern = rf'^(\s*)(?:async\s+)?def\s+{re.escape(element_name)}\s*\('
        
        for i, line in enumerate(lines):
            match = re.match(func_pattern, line)
            if match:
                indent = len(match.group(1))
                start_line = i + 1
                
                # Find end of function
                end_line = len(lines)
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if next_line.strip():
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent and (next_line.strip().startswith('def ') or 
                                                      next_line.strip().startswith('class ') or
                                                      next_line.strip().startswith('@')):
                            end_line = j
                            break
                
                code = '\n'.join(lines[i:end_line])
                
                return {
                    "success": True,
                    "file": file_path,
                    "element_name": element_name,
                    "element_type": "method" if element_name == '__init__' else "method",
                    "start_line": start_line,
                    "end_line": end_line,
                    "code": code,
                    "total_lines": len(lines)
                }
        
        # Try to find field/variable assignment
        field_pattern = rf'^\s*{re.escape(element_name)}\s*[=:]'
        
        for i, line in enumerate(lines):
            if re.match(field_pattern, line):
                return {
                    "success": True,
                    "file": file_path,
                    "element_name": element_name,
                    "element_type": "field",
                    "line_number": i + 1,
                    "code": line.strip(),
                    "total_lines": len(lines)
                }
        
        return {
            "success": False,
            "error": f"Element '{element_name}' not found in {file_path} (regex fallback)",
            "file": file_path,
            "element_name": element_name
        }
    
    def get_call_graph(self, file_path: str, function_name: str) -> Dict:
        """Get call graph for function"""
        try:
            code_result = self.extract_code(file_path)
            
            if not code_result.get("success"):
                return code_result
            
            code = code_result["code"]
            
            # Parse to find function
            tree = ast.parse(code)
            
            target_function = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == function_name:
                        target_function = node
                        break
            
            if not target_function:
                return {
                    "success": False,
                    "error": f"Function {function_name} not found in {file_path}"
                }
            
            # Extract calls within function
            callees = []
            for node in ast.walk(target_function):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        callees.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        callees.append(node.func.attr)
            
            return {
                "success": True,
                "file": file_path,
                "function": function_name,
                "callees": list(set(callees)),
                "num_calls": len(callees)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Call graph analysis failed: {str(e)}"
            }
    
    def analyze_dependencies(self, file_path: str, function_name: str) -> Dict:
        """Analyze field dependencies"""
        try:
            code_result = self.extract_code(file_path)
            
            if not code_result.get("success"):
                return code_result
            
            code = code_result["code"]
            tree = ast.parse(code)
            
            target_function = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == function_name:
                        target_function = node
                        break
            
            if not target_function:
                return {
                    "success": False,
                    "error": f"Function {function_name} not found"
                }
            
            reads = set()
            writes = set()
            imports = set()
            
            for node in ast.walk(target_function):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Load):
                        reads.add(node.id)
                    elif isinstance(node.ctx, ast.Store):
                        writes.add(node.id)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.ctx, ast.Load):
                        reads.add(node.attr)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        imports.add(alias.name)
            
            return {
                "success": True,
                "file": file_path,
                "function": function_name,
                "reads": sorted(list(reads)),
                "writes": sorted(list(writes)),
                "imports": sorted(list(imports))
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Dependency analysis failed: {str(e)}"
            }
    
    def find_api_usage(self, api_name: str) -> Dict:
        """Find API usage examples in codebase"""
        examples = []
        
        try:
            # Use grep to find usages
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", api_name, str(self.repo_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            for line in result.stdout.split('\n')[:10]:
                if line.strip():
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = parts[1]
                        code_line = parts[2]
                        
                        try:
                            rel_path = str(Path(file_path).relative_to(self.repo_path))
                        except ValueError:
                            rel_path = file_path
                        
                        examples.append({
                            "file": rel_path,
                            "line": line_num,
                            "code": code_line.strip()
                        })
            
            return {
                "success": True,
                "api_name": api_name,
                "num_usages": len(examples),
                "examples": examples
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"API usage search failed: {str(e)}",
                "examples": []
            }

# ============================================================================
# LLM Agents
# ============================================================================

class ContextUpdaterAgent:
    """Agent for dynamic context management"""
    
    SYSTEM_PROMPT = """You are a Context Manager Agent for automated program repair.

Your role is to analyze failed repair attempts and decide what additional information to retrieve.

Available tools:
1. run_coverage(full_test_name) - Execute test with coverage (full test name example: astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6])
2. search_similar_methods(code, k) - Find similar code
3. extract_code(file_path, element_name) - Get specific element by name (method/field/constructor)
   - Use element_name to extract: methods, test methods, fields, or __init__ constructors
   - Example: extract_code("utils.py", "validate_input") extracts the validate_input method
4. get_call_graph(file_path, function) - Analyze function calls
5. analyze_dependencies(file_path, function) - Check dependencies
6. find_api_usage(api_name) - Find API examples

Respond with JSON:
{
    "tools_to_execute": [
        {"tool": "tool_name", "params": {...}},
        ...
    ],
    "reasoning": "Why these tools will help"
}"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=config.openai_api_key)
    
    def decide_actions(self, context_pool: ContextPool, iteration: int) -> List[Dict]:
        """Decide what context to gather"""
        
        prompt = self._build_prompt(context_pool, iteration)
        
        try:
            response = self.client.chat.completions.create(
                model=config.model_context_updater,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temp_context_updater,
                max_tokens=config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("tools_to_execute", [])
            
        except Exception as e:
            print(f"  Error in ContextUpdater: {e}")
            return []
    
    def _build_prompt(self, context_pool: ContextPool, iteration: int) -> str:
        """Build decision prompt with clear decision → result mapping"""
        
        prompt = f"# Iteration {iteration}\n\n"
        
        prompt += "## Problem Statement\n"
        prompt += context_pool.problem_statement[:500] + "\n\n"
        
        prompt += "## Buggy Methods\n"
        for method_key, method_code in list(context_pool.buggy_methods.items())[:3]:
            prompt += f"### {method_key}\n```python\n{method_code}\n```\n\n"
        
        prompt += "## Failing Tests\n"
        for test in context_pool.failing_tests[:5]:
            prompt += f"- {test}\n"
        prompt += "\n"
        
        # 🆕 PREVIOUSLY MADE DECISIONS AND THEIR RESULTS
        if iteration > 1 and context_pool.iteration_logs:
            prompt += "## Previously Made Decisions and Results\n"
            prompt += "Review what tools were requested and what happened:\n\n"
            
            for prev_log in context_pool.iteration_logs:
                prompt += f"### Iteration {prev_log.iteration}\n"
                
                # Show what tools were decided
                if hasattr(prev_log, 'tool_executions') and prev_log.tool_executions:
                    prompt += "**Decisions Made:**\n"
                    
                    for idx, tool_exec in enumerate(prev_log.tool_executions, 1):
                        tool_name = tool_exec.get('tool', 'unknown')
                        params = tool_exec.get('params', {})
                        
                        # Format the decision
                        prompt += f"{idx}. {tool_name}(\n"
                        for key, value in params.items():
                            if isinstance(value, str) and len(value) > 50:
                                value = value[:50] + "..."
                            prompt += f"     {key}={repr(value)}\n"
                        prompt += "   )\n"
                    
                    prompt += "\n**Tool Execution Results:**\n"
                    
                    for idx, tool_exec in enumerate(prev_log.tool_executions, 1):
                        tool_name = tool_exec.get('tool', 'unknown')
                        success = tool_exec.get('success', False)
                        result = tool_exec.get('result')
                        error = tool_exec.get('error')
                        
                        status_icon = "✓" if success else "✗"
                        prompt += f"{idx}. {tool_name}: {status_icon} {'SUCCESS' if success else 'FAILED'}\n"
                        
                        if success and result:
                            # Extract meaningful info from result
                            if isinstance(result, dict):
                                prompt += "   Retrieved:\n"
                                
                                # Common result fields
                                if 'code' in result:
                                    code_len = len(result['code']) if isinstance(result['code'], str) else 0
                                    prompt += f"   - Code: {code_len} characters\n"
                                
                                if 'num_results' in result:
                                    prompt += f"   - Found: {result['num_results']} results\n"
                                
                                if 'results' in result and isinstance(result['results'], list):
                                    prompt += f"   - Results: {len(result['results'])} items\n"
                                    # Show first result as example
                                    if result['results']:
                                        first = result['results'][0]
                                        if isinstance(first, dict):
                                            if 'file' in first:
                                                prompt += f"     Example: {first.get('file')} - {first.get('name', 'N/A')}\n"
                                
                                if 'callees' in result:
                                    callees = result['callees'][:5]
                                    prompt += f"   - Callees: {', '.join(callees)}\n"
                                
                                if 'executed_lines' in result:
                                    files = list(result['executed_lines'].keys())[:3]
                                    prompt += f"   - Coverage: {len(result['executed_lines'])} files\n"
                                    if files:
                                        prompt += f"     Files: {', '.join(files)}\n"
                                
                                if 'element_type' in result:
                                    prompt += f"   - Element: {result['element_type']}\n"
                                
                                if 'reads' in result or 'writes' in result:
                                    reads = result.get('reads', [])[:5]
                                    writes = result.get('writes', [])[:5]
                                    if reads:
                                        prompt += f"   - Reads: {', '.join(reads)}\n"
                                    if writes:
                                        prompt += f"   - Writes: {', '.join(writes)}\n"
                            else:
                                prompt += f"   Retrieved: {str(result)[:200]}\n"
                        
                        elif not success:
                            prompt += f"   Error: {error[:200] if error else 'Unknown error'}\n"
                        
                        prompt += "\n"
                
                else:
                    prompt += "No tool decisions made in this iteration.\n\n"
                
                prompt += "---\n\n"
        
        # Show accumulated context
        if context_pool.dynamic_context:
            prompt += "## Context Already Retrieved\n"
            prompt += "The following information is already available from previous tool executions:\n"
            for ctx in context_pool.dynamic_context[-5:]:
                prompt += f"- [Iteration {ctx.iteration}] {ctx.source}\n"
            prompt += "\n"
        
        if context_pool.error_messages:
            prompt += "## Error Messages from Failed Patches\n"
            for err in context_pool.error_messages[:2]:
                prompt += f"```\n{err[:300]}\n```\n\n"
        
        if context_pool.tried_hypotheses:
            prompt += "## Previous Failed Repair Attempts\n"
            for h in context_pool.tried_hypotheses[-3:]:
                prompt += f"- {h.summary}\n"
                if h.test_results:
                    prompt += f"  Status: {h.test_results.get('status', 'unknown')}\n"
                    if 'error' in h.test_results:
                        prompt += f"  Error: {h.test_results['error'][:150]}\n"
            prompt += "\n"
        
        prompt += "\n## Task\n"
        prompt += "Based on the previously made decisions and their results above:\n"
        prompt += "1. Identify what information is MISSING that would help create a correct patch\n"
        prompt += "2. DO NOT repeat tool calls that already succeeded (the context is already available)\n"
        prompt += "3. DO NOT repeat tool calls that failed with the same parameters\n"
        prompt += "4. Request NEW tools or the SAME tools with DIFFERENT parameters to get new insights\n"
        prompt += "\nWhat NEW context should be retrieved for the next repair attempt?"
        
        return prompt


class GeneratorAgent:
    """Agent for patch generation"""
    
    SYSTEM_PROMPT = """You are a Patch Generator Agent for automated program repair.

Your role is to generate correct patches based on:
- Problem description
- Buggy methods (the actual code that needs fixing)
- Failing tests and errors
- Retrieved context
- Previous failed attempts

Respond with JSON containing a single patch candidate:
{
    "hypothesis": "Brief explanation of your fix strategy",
    "fixed_methods": {
        "file_path::method_name": "complete fixed method code",
        ...
    }
}

IMPORTANT: 
1. Fix ONLY the buggy methods provided
2. Return complete method implementations (including def line and full body)
3. Use the exact key format: "file_path::method_name"
4. Focus on one specific repair strategy per response"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=config.openai_api_key)
    
    def generate_patches(self, context_pool: ContextPool, iteration: int, test_codes, num_patches: int = 2) -> List[Hypothesis]:
        """Generate multiple patch hypotheses using n parameter"""
        prompt = self._build_prompt(context_pool, iteration = iteration, test_codes = test_codes, num_patches = num_patches)
        
        try:
            # Use n parameter to get multiple diverse completions
            response = self.client.chat.completions.create(
                model=config.model_generator,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temp_generator,
                max_tokens=config.max_tokens,
                n=5,  # Generate num_patches different completions
                response_format={"type": "json_object"}
            )
            
            hypotheses = []
            
            # Process each completion
            for i, choice in enumerate(response.choices, 1):
                try:
                    result = json.loads(choice.message.content)
                    
                    # Extract the single patch from this completion
                    fixed_methods = result.get("fixed_methods", {})
                    fixed_files = self._methods_to_files(fixed_methods, context_pool)
                    
                    hypothesis = Hypothesis(
                        summary=f"[{i}/{num_patches}] {result.get('hypothesis', '')}",
                        diff="",
                        fixed_code=fixed_files,
                        iteration=iteration
                    )
                    
                    # Compute semantic hash
                    code_str = json.dumps(hypothesis.fixed_code, sort_keys=True)
                    hypothesis.semantic_hash = hashlib.md5(code_str.encode()).hexdigest()
                    
                    hypotheses.append(hypothesis)
                    
                except json.JSONDecodeError as je:
                    print(f"  ⚠️  Failed to parse completion {i}: {je}")
                    continue
            
            print(f"  ✅ Generated {len(hypotheses)} patch candidates")
            return hypotheses
            
        except Exception as e:
            print(f"  ❌ Error in Generator: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _build_prompt(self, context_pool: ContextPool, iteration: int, test_codes, num_patches: int) -> str:
        """Build generation prompt"""
        
        prompt = f"# Repair Generation - Iteration {iteration}\n\n"
        
        prompt += "## Problem Statement\n"
        prompt += context_pool.problem_statement[:800] + "\n\n"
        
        # CRITICAL: Include buggy methods
        prompt += "## Buggy Methods (CODE TO FIX)\n"
        if context_pool.buggy_methods:
            for method_key, method_code in list(context_pool.buggy_methods.items())[:5]:
                prompt += f"### {method_key}\n```python\n{method_code}\n```\n\n"
        else:
            prompt += "### Original Code\n"
            for file, code in list(context_pool.original_code.items())[:2]:
                prompt += f"File: {file}\n```python\n{code[:800]}\n```\n\n"
        
        prompt += "## Failing Tests\n"
        for test in context_pool.failing_tests[:5]:
            prompt += f"- {test}\n"
        
        prompt += str(test_codes) + "\n"
        
        if context_pool.error_messages:
            prompt += "## Error Messages\n"
            for err in context_pool.error_messages[:3]:
                prompt += f"```\n{err[:300]}\n```\n\n"
        
        if context_pool.tried_hypotheses:
            prompt += "## Previous Failed Attempts\n"
            for h in context_pool.tried_hypotheses[-5:]:
                prompt += f"- {h.summary}\n"
            prompt += "\n"
        
        if context_pool.dynamic_context:
            prompt += "## Retrieved Context\n"
            for ctx in context_pool.dynamic_context[-5:]:
                prompt += f"[{ctx.source}] {ctx.content[:200]}\n"
            prompt += "\n"
        
        prompt += f"\n## Task\n"
        prompt += "Generate ONE patch candidate that fixes the buggy methods shown above.\n"
        prompt += "Focus on a specific repair strategy (e.g., null checks, type validation, edge cases, logic fixes).\n"
        prompt += "Use the key format: 'file_path::method_name' for each fixed method."
        
        return prompt
    
    def _methods_to_files(self, fixed_methods: Dict[str, str], context_pool: ContextPool) -> Dict[str, str]:
        """Convert fixed methods to complete files"""
        fixed_files = {}
        
        # Group methods by file
        methods_by_file = {}
        for method_key, fixed_method_code in fixed_methods.items():
            if '::' in method_key:
                file_path, method_name = method_key.rsplit('::', 1)
            else:
                # Fallback: try to find file from buggy_methods
                file_path = None
                method_name = method_key
                for buggy_key in context_pool.buggy_methods.keys():
                    if buggy_key.endswith(f'::{method_name}'):
                        file_path = buggy_key.rsplit('::', 1)[0]
                        break
                
                if not file_path:
                    print(f"  Warning: Could not determine file for method {method_name}")
                    continue
            
            if file_path not in methods_by_file:
                methods_by_file[file_path] = {}
            methods_by_file[file_path][method_name] = fixed_method_code
        
        # For each file, replace the buggy methods with fixed ones
        for file_path, methods in methods_by_file.items():
            original_content = context_pool.original_code.get(file_path, "")
            
            if not original_content:
                print(f"  Warning: No original content for {file_path}")
                continue
            
            # Replace methods in the file
            fixed_content = self._replace_methods_in_file(
                original_content, 
                methods,
                file_path
            )
            
            fixed_files[file_path] = fixed_content
        
        return fixed_files
    
    def _replace_methods_in_file(self, file_content: str, methods: Dict[str, str], file_path: str) -> str:
        """Replace methods in file content with fixed versions"""
        
        try:
            tree = ast.parse(file_content)
            lines = file_content.split('\n')
            
            # Find all methods to replace with their line ranges
            replacements = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = node.name
                    
                    if method_name in methods:
                        start_line = node.lineno - 1  # 0-indexed
                        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 50
                        
                        # Get the indentation of the original method
                        original_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
                        
                        # Get the fixed method code and adjust indentation
                        fixed_method = methods[method_name]
                        fixed_lines = fixed_method.split('\n')
                        
                        # Calculate indentation of fixed method
                        if fixed_lines:
                            fixed_indent = len(fixed_lines[0]) - len(fixed_lines[0].lstrip())
                            indent_diff = original_indent - fixed_indent
                            
                            # Adjust indentation
                            if indent_diff != 0:
                                adjusted_lines = []
                                for line in fixed_lines:
                                    if line.strip():  # Don't indent empty lines
                                        adjusted_lines.append(' ' * indent_diff + line)
                                    else:
                                        adjusted_lines.append(line)
                                fixed_method = '\n'.join(adjusted_lines)
                        
                        replacements.append((start_line, end_line, fixed_method))
            
            # Apply replacements in reverse order to maintain line numbers
            replacements.sort(reverse=True)
            
            for start_line, end_line, fixed_method in replacements:
                # Replace the lines
                lines[start_line:end_line] = fixed_method.split('\n')
            
            return '\n'.join(lines)
            
        except SyntaxError as e:
            print(f"  Warning: Could not parse {file_path} for method replacement: {e}")
            # Fallback: use regex-based replacement
            return self._replace_methods_regex(file_content, methods)
    
    def _replace_methods_regex(self, file_content: str, methods: Dict[str, str]) -> str:
        """Fallback: Replace methods using regex"""
        
        result = file_content
        
        for method_name, fixed_method in methods.items():
            # Pattern to match the method definition and its body
            pattern = rf'(^[ \t]*)(?:async\s+)?def\s+{re.escape(method_name)}\s*\([^)]*\).*?(?=\n\1(?:def|class|$)|\Z)'
            
            # Try to replace
            new_result = re.sub(pattern, fixed_method, result, flags=re.MULTILINE | re.DOTALL)
            
            if new_result != result:
                result = new_result
            else:
                print(f"  ⚠ Could not replace method {method_name}")
        
        return result


class OverfittingDetectorAgent:
    """Agent for detecting and preventing overfitting"""
    
    SYSTEM_PROMPT = """You are an Overfitting Detection Agent.

Analyze patches for overfitting patterns:
1. Hard-coded test values
2. Overly specific conditions
3. Test-dependent logic
4. Missing edge case handling
5. Superficial fixes

Respond with JSON:
{
    "is_overfitting": true/false,
    "confidence": 0.0-1.0,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"]
}"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=config.openai_api_key)
    
    def check_overfitting(self, hypothesis: Hypothesis, context_pool: ContextPool) -> Dict:
        """Check if patch is overfitting"""
        
        prompt = self._build_prompt(hypothesis, context_pool)
        
        try:
            response = self.client.chat.completions.create(
                model=config.model_overfitting,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temp_overfitting,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"  Error in OverfittingDetector: {e}")
            return {"is_overfitting": False, "confidence": 0.0}
    
    def _build_prompt(self, hypothesis: Hypothesis, context_pool: ContextPool) -> str:
        """Build detection prompt"""
        
        prompt = "## Patch to Analyze\n"
        prompt += f"Strategy: {hypothesis.summary}\n\n"
        
        prompt += "## Fixed Code\n"
        for file, code in hypothesis.fixed_code.items():
            prompt += f"File: {file}\n```python\n{code[:1000]}\n```\n\n"
        
        prompt += "## Failing Tests\n"
        for test in context_pool.failing_tests[:5]:
            prompt += f"- {test}\n"
        
        prompt += "\n## Task\nCheck if this patch is overfitting to the tests."
        
        return prompt

# ============================================================================
# SWE-bench Evaluation Integration
# ============================================================================

class SWEBenchEvaluator:
    """Integration with official SWE-bench evaluation harness"""
    
    def __init__(self, predictions_dir: str):
        self.predictions_dir = Path(predictions_dir)
        self.predictions_dir.mkdir(exist_ok=True, parents=True)
        
    def create_prediction_file(self, instance_id: str, model_patch: str, patch_id: str) -> Path:
        """Create a separate prediction file for a single patch"""
        
        # Validate patch format
        if not self._validate_patch(model_patch):
            print(f"    ⚠️  Warning: Invalid patch format, skipping")
            return None
        
        # Create subdirectory for this bug
        bug_dir = self.predictions_dir / instance_id
        bug_dir.mkdir(exist_ok=True, parents=True)
        
        # Create prediction file for this specific patch
        pred_file = bug_dir / f"patch_{patch_id}.jsonl"
        
        prediction = {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": f"agent_apr_system_patch_{patch_id}"
        }
        
        with open(pred_file, 'w') as f:
            f.write(json.dumps(prediction) + '\n')
        
        return pred_file
    
    def _validate_patch(self, patch: str) -> bool:
        """Validate patch format"""
        if not patch or not patch.strip():
            return False
        
        # Check for required diff headers
        if 'diff --git' not in patch:
            return False
        if '---' not in patch or '+++' not in patch:
            return False
        if '@@' not in patch:  # Hunk header
            return False
        
        # Check for changes (+ or - lines)
        has_changes = False
        for line in patch.split('\n'):
            if line.startswith('+') or line.startswith('-'):
                if not line.startswith('+++') and not line.startswith('---'):
                    has_changes = True
                    break
        
        return has_changes

    def evaluate_single_patch(self, pred_file: Path, instance_id: str) -> dict:
        """Evaluate a single patch using SWE-bench official harness."""

        # Read prediction to get model name
        with open(pred_file, "r") as f:
            line = f.readline()
            pp = json.loads(line) if line else {}
        
        model = pp.get("model_name_or_path", "unknown_model")
        run_id = f"{instance_id}_{pred_file.stem}"
        
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", "princeton-nlp/SWE-bench_Lite",
            "--split", "test",
            "--instance_ids", instance_id,
            "--predictions_path", str(pred_file),
            "--max_workers", "1",
            "--timeout", "900",
            "--cache_level", "instance",
            "--run_id", run_id,
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            results_file = f"/home/selab/Desktop/swe_exp/logs/run_evaluation/{run_id}/{model}/{instance_id}/report.json"
            test_output_file = f"/home/selab/Desktop/swe_exp/logs/run_evaluation/{run_id}/{model}/{instance_id}/test_output.txt"

            # Extract error messages from test output
            error_messages = self._extract_error_messages(test_output_file)

            if os.path.exists(results_file):
                with open(results_file) as f:
                    data = json.load(f)

                if isinstance(data, list) and data:
                    resolved = bool(data[0].get("resolved", False))
                    return {
                        "resolved": resolved,
                        "status": "plausible" if resolved else "failed",
                        "run_id": run_id,
                        "details": data[0],
                        "error_messages": error_messages
                    }
                elif isinstance(data, dict):
                    data = data.get(instance_id, {})
                    resolved = bool(data.get("resolved", False))
                    return {
                        "resolved": resolved,
                        "status": "plausible" if resolved else "failed",
                        "run_id": run_id,
                        "details": data,
                        "error_messages": error_messages
                    }

            return {
                "resolved": False,
                "status": "failed",
                "error": "No report.json found",
                "error_messages": error_messages,
                "stdout": result.stdout[-500:],
                "stderr": result.stderr[-500:]
            }

        except subprocess.TimeoutExpired:
            return {"resolved": False, "status": "timeout", "error": "Evaluation timed out"}

        except Exception as e:
            import traceback
            return {
                "resolved": False, 
                "status": "error", 
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        
    def _extract_error_messages(self, test_output_file: str) -> List[str]:
        """Extract error messages from test output file"""
        error_messages = []
        
        if not os.path.exists(test_output_file):
            return error_messages
        
        try:
            with open(test_output_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract FAILED tests summary
            if 'FAILED' in content:
                failed_lines = [line for line in content.split('\n') if 'FAILED' in line and '::' in line]
                for line in failed_lines[:10]:
                    error_messages.append(f"Failed: {line.strip()}")
            
            # Look for FAILURES section with detailed errors
            if '=== FAILURES ===' in content or 'FAILURES' in content:
                # Split by test case failures
                failure_marker = '___ '
                parts = content.split(failure_marker)
                
                for i, part in enumerate(parts[1:], 1):  # Skip first part (before failures)
                    if i > 5:  # Limit to 5 test failures
                        break
                    
                    # Extract test name (first line after marker)
                    lines = part.split('\n')
                    if lines:
                        test_name = lines[0].strip('_ ')
                        
                        # Find assertion error or exception
                        error_section = []
                        capture = False
                        
                        for line in lines:
                            # Start capturing at assertion, exception, or error marker
                            if any(marker in line for marker in ['>       ', 'E       ', 'AssertionError', 'Error:', 'Exception:']):
                                capture = True
                            
                            # Stop at next section marker
                            if capture and ('===' in line or '___' in line):
                                break
                            
                            if capture and line.strip():
                                error_section.append(line)
                        
                        if error_section:
                            error_msg = f"\nTest: {test_name}\n" + '\n'.join(error_section[:25])  # Limit lines
                            error_messages.append(error_msg)
            
            # Extract short test summary if available
            if 'short test summary' in content.lower():
                summary_start = content.lower().find('short test summary')
                summary_end = content.find('===', summary_start + 100)
                if summary_start != -1:
                    if summary_end != -1:
                        summary = content[summary_start:summary_end].strip()
                    else:
                        summary = content[summary_start:summary_start+800].strip()
                    if summary and len(summary) < 2000:
                        error_messages.append(f"\n{summary}")
            
            # Extract ERROR sections (setup/teardown errors)
            if '=== ERRORS ===' in content:
                error_start = content.find('=== ERRORS ===')
                error_end = content.find('===', error_start + 20)
                if error_end == -1:
                    error_end = error_start + 1000
                error_section = content[error_start:error_end]
                if error_section:
                    error_messages.append(f"\nSetup/Teardown Errors:\n{error_section[:800]}")
                        
        except Exception as e:
            error_messages.append(f"Error reading test output: {str(e)}")
        
        return error_messages[:10]  # Limit total error messages


# ============================================================================
# Repair Pipeline (Updated)
# ============================================================================

class RepairPipeline:
    """Main repair orchestration with multi-patch generation"""
    
    def __init__(self, bug: BugInstance, repo_path: str):
        self.bug = bug
        self.repo_path = Path(repo_path)
        self.tools = ToolSuite(repo_path, bug.instance_id)
        
        self.context_updater = ContextUpdaterAgent()
        self.generator = GeneratorAgent()
        self.overfitting_detector = OverfittingDetectorAgent()
        
        self.context_pool = ContextPool()
        self.evaluator = SWEBenchEvaluator(config.predictions_dir) if config.use_swebench_eval else None
        
    def initialize(self):
        """Initialize context pool and vector DB"""
        print(f"  Initializing repair for {self.bug.instance_id}")
        
        # Load original code
        self._load_original_code()
        
        # Extract buggy methods from patch
        print(f"  Extracting buggy methods from patch...")
        self.context_pool.buggy_methods = BuggyMethodExtractor.extract_from_patch(
            self.bug.patch,
            self.repo_path
        )
        print(f"  Found {len(self.context_pool.buggy_methods)} buggy methods")
        
        # Setup context pool
        self.context_pool.problem_statement = self.bug.problem_statement
        self.context_pool.failing_tests = self.bug.fail_to_pass
        
    def _load_original_code(self):
        """Load buggy code files"""
        import re
        files = re.findall(r'---\s+a/(.*?)\s+\+\+\+', self.bug.patch)
        
        for file in files[:5]:
            # Extract full file (no element_name specified)
            code_result = self.tools.extract_code(file)
            if code_result.get("success"):
                self.context_pool.original_code[file] = code_result["code"]
    
    def run(self) -> Dict:
        """Execute repair pipeline - evaluate each patch immediately"""
        start_time = time.time()
        plausible_patch = None
        
        for iteration in range(1, config.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{config.max_iterations}")
            print(f"{'='*60}")
            
            # Step 1: Context Update
            tool_execution_logs = []  # ADD THIS LINE
            
            if iteration > 1:
                tools_to_run = self.context_updater.decide_actions(
                    self.context_pool, iteration
                )
                tool_execution_logs = self._execute_tools(tools_to_run, iteration)  # CHANGE THIS LINE
            
            # Step 2: Generate Multiple Patches
            print(f"  🎯 Generating 10 patch candidates...")
            test_codes = ""
            for test in self.context_pool.failing_tests:
                test_code_result = self.tools.extract_code(test.split("::")[0], test.split("::")[1] if "::" in test else None)
                if test_code_result.get("success"):
                    test_codes += f"{test_code_result['code']}\n\n"
            # test_codes+= "Failure detection code (e.g., assertions):\n"
            # test_codes+= self.bug.test_patch
            
            hypotheses = self.generator.generate_patches(
                self.context_pool, iteration, test_codes, num_patches=10
            )
            
            if not hypotheses:
                print("  ❌ Failed to generate hypotheses")
                continue
            
            prompt = self.generator._build_prompt(self.context_pool, iteration, test_codes, 10)
            
            # Step 3: Evaluate Each Patch with SWE-bench
            iteration_results = []
   
            for idx, hypothesis in enumerate(hypotheses, 1):
                patch_id = f"iter{iteration}_patch{idx}"
                print(f"\n  [{idx}/10] Evaluating: {hypothesis.summary}")
                
                # Create unified diff
                model_patch = self._create_unified_diff(hypothesis)
                
                if not model_patch:
                    print(f"    ❌ Failed to create valid patch - skipping")
                    continue
                
                hypothesis.model_patch = model_patch
                
                # Create separate prediction file for this patch
                if self.evaluator:
                    pred_file = self.evaluator.create_prediction_file(
                        self.bug.instance_id,
                        model_patch,
                        patch_id
                    )
                    
                    if not pred_file:
                        print(f"    ❌ Failed to create prediction file - skipping")
                        continue
                    
                    print(f"    📝 Created: {pred_file}")
                    
                    # Evaluate this patch with SWE-bench
                    print(f"    🔬 Running SWE-bench evaluation...")
                    eval_result = self.evaluator.evaluate_single_patch(
                        pred_file,
                        self.bug.instance_id
                    )
                    
                    hypothesis.test_results = eval_result
                    
                    iteration_results.append({
                        'patch_id': patch_id,
                        'summary': hypothesis.summary,
                        'semantic_hash': hypothesis.semantic_hash,
                        'eval_result': eval_result if self.evaluator else {'status': 'not_evaluated'},
                        'pred_file': str(pred_file) if self.evaluator else None
                    })
                    # Check if plausible (passes tests)
                    if eval_result.get("resolved", False):
                        print(f"    ✅ PLAUSIBLE! Patch passes tests")
                        
                        # Check overfitting
                        print(f"    🔍 Checking for overfitting...")
                        overfitting_check = self.overfitting_detector.check_overfitting(
                            hypothesis, self.context_pool
                        )
                        
                        if overfitting_check.get("is_overfitting", False):
                            confidence = overfitting_check.get('confidence', 0)
                            print(f"    ⚠️  Overfitting detected (confidence: {confidence:.2f})")
                            print(f"    🔄 Refining patch...")
                            
                            # Refine once
                            refined_hypothesis = self._refine_overfitting_patch(
                                hypothesis,
                                overfitting_check,
                                iteration
                            )
                            
                            if refined_hypothesis:
                                # Evaluate refined patch
                                refined_patch_id = f"{patch_id}_refined"
                                refined_model_patch = self._create_unified_diff(refined_hypothesis)
                                refined_hypothesis.model_patch = refined_model_patch
                                if not refined_model_patch or 'diff --git' not in refined_model_patch:
                                    print(f"    ❌ Failed to create valid refined patch - keeping original")
                                    plausible_patch = hypothesis
                                else:
                                    refined_pred_file = self.evaluator.create_prediction_file(
                                        self.bug.instance_id,
                                        refined_model_patch,
                                        refined_patch_id
                                    )
                                    
                                    print(f"    🔬 Evaluating refined patch...")
                                    refined_eval = self.evaluator.evaluate_single_patch(
                                        refined_pred_file,
                                        self.bug.instance_id
                                    )
                                    
                                    refined_hypothesis.test_results = refined_eval
                                    
                                    if refined_eval.get("resolved", False):
                                        print(f"    ✅ Refined patch is plausible!")
                                        plausible_patch = refined_hypothesis
                                    else:
                                        print(f"    ❌ Refined patch failed")
                                        # Keep original plausible patch
                                        plausible_patch = hypothesis
                            else:
                                # Keep original if refinement failed
                                plausible_patch = hypothesis
                        else:
                            # No overfitting, this is our patch!
                            print(f"    ✅ No overfitting detected!")
                            plausible_patch = hypothesis   
                        # Found plausible patch, stop iteration
                        self.context_pool.tried_hypotheses.append(plausible_patch)
                        break
                    else:
                        print(f"    ❌ Failed: {eval_result.get('error', 'Tests did not pass')}")
                
                # Update history
                self.context_pool.tried_hypotheses.append(hypothesis)
            
            # Step 4: Save Iteration Log
            iteration_log = IterationLog(
                iteration=iteration,
                timestamp=time.time(),
                prompt=prompt,
                hypotheses=iteration_results,
                error_messages=[],
                success=plausible_patch is not None,
                successful_hypothesis={
                    'summary': plausible_patch.summary,
                    'patch_id': iteration_results[[h['summary'] for h in iteration_results].index(plausible_patch.summary)]['patch_id'],
                    'fixed_code': plausible_patch.fixed_code,
                    'overfitting_refined': 'refined' in iteration_results[[h['summary'] for h in iteration_results].index(plausible_patch.summary)]['patch_id']
                } if plausible_patch else None,
                tool_executions=tool_execution_logs  # ADD THIS LINE
            )
            
            self.context_pool.iteration_logs.append(iteration_log)
            self._save_iteration_log(iteration_log)
            
            # Step 5: Check if we found plausible patch
            if plausible_patch:
                print(f"\n{'='*60}")
                print(f"✅ PLAUSIBLE PATCH FOUND!")
                print(f"{'='*60}")
                print(f"Iteration: {iteration}")
                print(f"Hypothesis: {plausible_patch.summary}")
                print(f"Time: {time.time() - start_time:.1f}s")
                
                return {
                    "success": True,
                    "iteration": iteration,
                    "hypothesis": plausible_patch.summary,
                    "model_patch": plausible_patch.model_patch,
                    "time": time.time() - start_time,
                    "total_patches_evaluated": len(self.context_pool.tried_hypotheses),
                    "eval_result": plausible_patch.test_results
                }
            
            print(f"\n  📊 Iteration {iteration} summary:")
            print(f"     Evaluated: {len(iteration_results)} patches")
            print(f"     All failed - moving to next iteration")
            
            if time.time() - start_time > config.timeout_per_bug:
                print("  ⏱️  Timeout reached")
                break
        
        # No plausible patch found
        print(f"\n{'='*60}")
        print(f"❌ No plausible patch found")
        print(f"{'='*60}")
        print(f"Total patches evaluated: {len(self.context_pool.tried_hypotheses)}")
        print(f"Time: {time.time() - start_time:.1f}s")
        
        return {
            "success": False,
            "iterations": config.max_iterations,
            "time": time.time() - start_time,
            "total_patches_evaluated": len(self.context_pool.tried_hypotheses)
        }
    
    def _refine_overfitting_patch(self, hypothesis: Hypothesis, 
                                  overfitting_check: Dict, iteration: int) -> Optional[Hypothesis]:
        """Refine an overfitting patch using the refinement agent"""
        
        # Build refinement prompt
        refinement_prompt = f"""# Patch Refinement - Iteration {iteration}

## Original Patch (Overfitting Detected)
Strategy: {hypothesis.summary}

## Overfitting Issues
Confidence: {overfitting_check.get('confidence', 0):.2f}
Issues: {', '.join(overfitting_check.get('issues', []))}
Suggestions: {', '.join(overfitting_check.get('suggestions', []))}

## Fixed Code
"""
        for file, code in hypothesis.fixed_code.items():
            refinement_prompt += f"File: {file}\n```python\n{code[:1000]}\n```\n\n"
        
        refinement_prompt += """
## Task
Refine this patch to address the overfitting issues while maintaining correctness.
Generate a better, more general solution.
"""
        
        try:
            response = self.generator.client.chat.completions.create(
                model=config.model_generator,
                messages=[
                    {"role": "system", "content": """You are refining an overfitting patch.
Generate a more general solution that addresses the overfitting issues.

Respond with JSON:
{
    "hypothesis": "Explanation of refined approach",
    "fixed_methods": {
        "file_path::method_name": "refined method code"
    }
}"""},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=config.temp_generator,
                max_tokens=config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            fixed_methods = result.get("fixed_methods", {})
            fixed_files = self.generator._methods_to_files(fixed_methods, self.context_pool)
            
            refined_hypothesis = Hypothesis(
                summary=f"[REFINED] {result.get('hypothesis', '')}",
                diff="",
                fixed_code=fixed_files,
                iteration=iteration
            )
            
            code_str = json.dumps(refined_hypothesis.fixed_code, sort_keys=True)
            refined_hypothesis.semantic_hash = hashlib.md5(code_str.encode()).hexdigest()
            
            return refined_hypothesis
            
        except Exception as e:
            print(f"    ⚠️  Refinement failed: {e}")
            return None
    
    def _save_iteration_log(self, iteration_log: IterationLog):
        """Save iteration log to file"""
        log_dir = Path(config.results_dir) / "iteration_logs" / self.bug.instance_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"iteration_{iteration_log.iteration}.json"
        
        log_data = {
            'iteration': iteration_log.iteration,
            'timestamp': iteration_log.timestamp,
            'prompt': iteration_log.prompt,
            'hypotheses': iteration_log.hypotheses,
            'error_messages': iteration_log.error_messages,
            'success': iteration_log.success,
            'successful_hypothesis': iteration_log.successful_hypothesis,
            'tool_executions': iteration_log.tool_executions  # ADD THIS LINE
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"    💾 Iteration log saved: {log_file}")
    
    def _validate_patch(self, hypothesis: Hypothesis) -> Dict:
        """Validate patch - to be implemented by subclass or integrated system"""
        return {
            "status": "not_implemented",
            "error": "Validation not implemented in base pipeline"
        }
    
    def _create_unified_diff(self, hypothesis: Hypothesis) -> str:
        """Create unified diff format patch for SWE-bench"""
        import difflib
        
        diff_parts = []
        
        for file_path, new_content in hypothesis.fixed_code.items():
            # Get original content
            original_content = self.context_pool.original_code.get(file_path, "")
            
            if not original_content:
                # Try to read from file
                code_result = self.tools.extract_code(file_path)
                if code_result.get("success"):
                    original_content = code_result["code"]
                else:
                    print(f"    ⚠️  Warning: Could not get original content for {file_path}")
                    continue
            
            # Ensure content ends with newline
            if original_content and not original_content.endswith('\n'):
                original_content += '\n'
            if new_content and not new_content.endswith('\n'):
                new_content += '\n'
            
            # Generate unified diff
            original_lines = original_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            
            # Create diff with proper headers
            diff = difflib.unified_diff(
                original_lines,
                new_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm=''
            )
            
            diff_lines = list(diff)
            
            if diff_lines:
                # Add proper git diff header
                git_header = f"diff --git a/{file_path} b/{file_path}\n"
                git_header += f"--- a/{file_path}\n"
                git_header += f"+++ b/{file_path}\n"
                
                # Combine header with diff content (skip the --- +++ from unified_diff)
                diff_content = []
                for i, line in enumerate(diff_lines):
                    if i < 2:  # Skip first two lines (--- and +++)
                        continue
                    if line.endswith('\n'):
                        diff_content.append(line)
                    else:
                        diff_content.append(line + '\n')
                
                if diff_content:
                    full_diff = git_header + ''.join(diff_content)
                    diff_parts.append(full_diff)
        
        if not diff_parts:
            print(f"    ⚠️  Warning: No diff generated")
            return ""
        
        # Join all file diffs
        final_patch = ''.join(diff_parts)
        
        # Ensure patch ends with newline
        if final_patch and not final_patch.endswith('\n'):
            final_patch += '\n'
        
        return final_patch
    
    def _execute_tools(self, tools: List[Dict], iteration: int) -> List[Dict]:
        """Execute requested tools with validation and return execution logs"""
        
        tool_execution_logs = []  # ADD THIS LINE
             
        for tool_req in tools[:5]:
            tool_name = tool_req.get("tool", "")
            params = tool_req.get("params", {})
            
            print(f"  🔧 Executing: {tool_name}")
            print(f"     Params: {params}")
            
            execution_log = {  # ADD THIS
                "tool": tool_name,
                "params": params,
                "timestamp": time.time(),
                "success": False,
                "result": None,
                "error": None
            }
            
            try:
                result = self.tools.execute_tool(tool_name, params)
                
                execution_log["success"] = result.get("success", False)  # ADD THIS
                execution_log["result"] = result  # ADD THIS
                
                if not result.get("success", False):
                    print(f"  ⚠️  Tool failed: {result.get('error', 'Unknown error')}")
                    execution_log["error"] = result.get("error", "Unknown error")  # ADD THIS
                    continue
                
                print(f"  ✅ Tool completed successfully")
                
                ctx_item = ContextItem(
                    content=json.dumps(result, indent=2),
                    source=tool_name,
                    timestamp=time.time(),
                    iteration=iteration,
                    relevance_score=1.0
                )
                self.context_pool.dynamic_context.append(ctx_item)
                
                if tool_name not in self.context_pool.tool_extractions:
                    self.context_pool.tool_extractions[tool_name] = []
                self.context_pool.tool_extractions[tool_name].append(result)
                
            except Exception as e:
                print(f"  ⚠️  Tool error: {e}")
                execution_log["error"] = str(e)  # ADD THIS
                import traceback
                traceback.print_exc()
            
            finally:  # ADD THIS BLOCK
                tool_execution_logs.append(execution_log)
        
        return tool_execution_logs  # ADD THIS LINE


# ============================================================================
# Main Execution Functions
# ============================================================================

def load_swebench_lite() -> List[BugInstance]:
    """Load SWE-bench Lite dataset"""
    dataset_path = Path("./datasets/swebench_lite.json")
    
    if not dataset_path.exists():
        print("⚠️  Dataset not found. Please run dataset download first.")
        return []
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    bugs = []
    for item in data:
        bug = BugInstance(
            instance_id=item["instance_id"],
            repo=item["repo"],
            base_commit=item["base_commit"],
            problem_statement=item["problem_statement"],
            hints_text=item.get("hints_text", ""),
            test_patch=item["test_patch"],
            patch=item["patch"],
            version=item["version"],
            fail_to_pass=item.get("FAIL_TO_PASS", []),
            pass_to_pass=item.get("PASS_TO_PASS", [])
        )
        bugs.append(bug)
    
    return bugs


def main():
    """Main execution"""
    print("="*70)
    print("Agent-Based Automated Program Repair System")
    print("SWE-bench Lite with Multi-Patch Generation")
    print("="*70)
    
    # Load dataset
    bugs = load_swebench_lite()
    print(f"\n✅ Loaded {len(bugs)} bugs from SWE-bench Lite")
    
    if not bugs:
        print("\n⚠️  No bugs loaded. Please download SWE-bench Lite dataset.")
        return
    
    # This is just the core system - use integrated_main.py for full execution
    print("\nThis is the core system module.")
    print("Please run: python integrated_main.py")


if __name__ == "__main__":
    main()