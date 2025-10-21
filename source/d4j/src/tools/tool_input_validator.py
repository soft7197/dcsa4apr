# src/tools/tool_input_validator.py
"""
Updated tool input validator with improved file matching specificity.
Fixed potential false positive file matches.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from fuzzywuzzy import fuzz, process


@dataclass
class ResolvedInput:
    """Represents a resolved and validated tool input."""
    tool_name: str
    params: Dict
    confidence: float  # 0.0 to 1.0
    resolution_notes: List[str]
    is_valid: bool


@dataclass
class ValidationResult:
    """Result of tool validation."""
    is_valid: bool
    corrected_params: Dict
    errors: List[str]
    warnings: List[str]
    confidence: float


class ProjectIndexer:
    """Index project structure for smart resolution."""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.logger = logging.getLogger(__name__)
        
        # Build indices
        self.file_index = {}
        self.class_index = {}
        self.method_index = {}
        self.test_index = {}
        self.file_path_index = {}  # Full path mapping
        
        self._build_indices()
    
    def _build_indices(self):
        """Build indices of project structure."""
        if not os.path.exists(self.project_path):
            self.logger.warning(f"Project path doesn't exist: {self.project_path}")
            return
        
        for root, dirs, files in os.walk(self.project_path):
            # Skip build and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['build', 'target', 'node_modules', '__pycache__']]
            
            # Limit depth
            depth = root[len(self.project_path):].count(os.sep)
            if depth > 5:
                continue
            
            for file in files:
                if file.endswith(('.java', '.py')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.project_path)
                    
                    # Index file with both basename and relative path
                    self.file_index[file] = file_path
                    self.file_path_index[relative_path] = file_path
                    
                    # Extract class name
                    class_name = file.rsplit('.', 1)[0]
                    
                    # Store with package/module path for uniqueness
                    package_path = relative_path.replace(os.sep, '.').rsplit('.', 1)[0]
                    full_class_name = f"{package_path}.{class_name}"
                    
                    self.class_index[class_name] = file_path
                    self.class_index[full_class_name] = file_path  # Also store full qualified name
                    
                    # Mark test files
                    if 'test' in file_path.lower() or file.startswith('Test') or file.endswith('Test.java') or file.endswith('_test.py'):
                        self.test_index[class_name] = file_path
                        self.test_index[full_class_name] = file_path
    
    def find_file_path(self, partial_path: str, prefer_test: bool = False) -> Optional[str]:
        """Find full file path from partial path with improved matching."""
        if not partial_path:
            return None
        
        # Direct match in file_path_index
        if partial_path in self.file_path_index:
            return self.file_path_index[partial_path]
        
        # Try with common source directories
        common_dirs = ['src/main/java', 'src/test/java', 'src', 'test', 'tests']
        for dir_prefix in common_dirs:
            test_path = os.path.join(dir_prefix, partial_path)
            if test_path in self.file_path_index:
                return self.file_path_index[test_path]
        
        # Try exact filename match
        basename = os.path.basename(partial_path)
        if basename in self.file_index:
            matches = [path for path in self.file_index.values() if os.path.basename(path) == basename]
            
            # Prefer test files if requested
            if prefer_test and len(matches) > 1:
                test_matches = [m for m in matches if 'test' in m.lower()]
                if test_matches:
                    return test_matches[0]
            
            # Return the match with the most similar directory structure
            if len(matches) > 1:
                partial_dir = os.path.dirname(partial_path)
                best_match = max(matches, key=lambda m: self._path_similarity(partial_dir, os.path.dirname(m)))
                return best_match
            
            return matches[0] if matches else None
        
        # Fuzzy match as last resort
        all_relative_paths = list(self.file_path_index.keys())
        best_match = process.extractOne(partial_path, all_relative_paths, scorer=fuzz.ratio)
        
        if best_match and best_match[1] > 80:  # Higher threshold for path matching
            return self.file_path_index[best_match[0]]
        
        return None
    
    def _path_similarity(self, path1: str, path2: str) -> float:
        """Calculate similarity between two paths."""
        parts1 = path1.split(os.sep)
        parts2 = path2.split(os.sep)
        
        # Count matching parts from the end (more specific)
        matching = 0
        for p1, p2 in zip(reversed(parts1), reversed(parts2)):
            if p1 == p2:
                matching += 1
            else:
                break
        
        max_len = max(len(parts1), len(parts2))
        return matching / max_len if max_len > 0 else 0
    
    def find_class_path(self, class_name: str) -> Optional[str]:
        """Find candidate file path for a class or interface by name."""
        # Direct match in index
        if class_name in self.class_index:
            return self.class_index[class_name]

        # Try fuzzy matching on simple names (no package)
        if '.' not in class_name:
            all_classes = [k for k in self.class_index.keys() if '.' not in k]
            best_match = process.extractOne(class_name, all_classes, scorer=fuzz.ratio)
            if best_match and best_match[1] > 85:
                return self.class_index[best_match[0]]

        return None

    
    def find_test_path(self, test_name: str) -> Optional[str]:
        """Find test file path."""
        # Direct match in test index
        if test_name in self.test_index:
            return self.test_index[test_name]
        
        # Try adding Test prefix/suffix
        variations = [
            f"Test{test_name}",
            f"{test_name}Test",
            f"{test_name}Tests",
            f"test_{test_name.lower()}",
            test_name.replace('test_', '').replace('Test', '')
        ]
        
        for variant in variations:
            if variant in self.test_index:
                return self.test_index[variant]
        
        # Fuzzy match in test files only
        if self.test_index:
            best_match = process.extractOne(test_name, list(self.test_index.keys()), scorer=fuzz.ratio)
            if best_match and best_match[1] > 80:
                return self.test_index[best_match[0]]
        
        return None
    
    def find_method_locations(self, method_name: str) -> List[Tuple[str, str, int]]:
        """Find all locations of a method. Returns (class_name, file_path, line_number)."""
        locations = []
        
        # Compile regex patterns for method detection
        java_pattern = re.compile(
            rf'\b(?:public|protected|private|static|\s)+[\w<>\[\]]+\s+{re.escape(method_name)}\s*\([^)]*\)\s*\{{',
            re.MULTILINE
        )
        python_pattern = re.compile(
            rf'^\s*def\s+{re.escape(method_name)}\s*\([^)]*\)\s*:',
            re.MULTILINE
        )
        
        for class_name, file_path in self.class_index.items():
            if '.' in class_name:  # Skip fully qualified names in search
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Search based on file extension
                if file_path.endswith('.java'):
                    for match in java_pattern.finditer(content):
                        line_number = content[:match.start()].count('\n') + 1
                        locations.append((class_name, file_path, line_number))
                
                elif file_path.endswith('.py'):
                    for match in python_pattern.finditer(content):
                        line_number = content[:match.start()].count('\n') + 1
                        locations.append((class_name, file_path, line_number))
                        
            except Exception as e:
                self.logger.debug(f"Error scanning {file_path}: {e}")
                continue
        
        return locations


class ToolInputValidator:
    """
    Validates and resolves tool inputs with improved accuracy.
    """
    
    def __init__(self, project_path: str = None, bug_info: Dict = None, project_indexer: ProjectIndexer = None):
        self.project_path = project_path
        self.bug_info = bug_info or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize or use provided project indexer
        self.indexer = project_indexer or (ProjectIndexer(project_path) if project_path else None)
        
        # Tool schemas
        self.tool_schemas = {
            'similar_method_search': {
                'required': ['method_body'],
                'optional': ['top_k', 'filter_keywords'],
                'defaults': {'top_k': 5, 'filter_keywords': []},
                'types': {'method_body': str, 'top_k': int, 'filter_keywords': list}
            },
            'code_extractor': {
                'required': ['element_name'],
                'optional': ['file_path', 'element_type', 'language'],
                'defaults': {'file_path': '', 'element_type': 'method', 'language': None},
                'types': {'file_path': str, 'element_name': str, 'element_type': str, 'language': str}
            },
            'field_dependency_analyzer': {
                'required': ['class_name', 'method_name'],
                'optional': ['file_path'],
                'defaults': {'file_path': None},
                'types': {'class_name': str, 'method_name': str, 'file_path': str}
            },
            'call_graph_builder': {
                'required': ['method_name'],
                'optional': ['file_path', 'direction', 'max_depth'],
                'defaults': {'file_path': None, 'direction': 'both', 'max_depth': 2},
                'types': {'method_name': str, 'file_path': str, 'direction': str, 'max_depth': int}
            },
            'api_usage_finder': {
                'required': ['api_class', 'api_method'],
                'optional': ['max_examples'],
                'defaults': {'max_examples': 5},
                'types': {'api_class': str, 'api_method': str, 'max_examples': int}
            },
            'coverage_runner': {
                'required': ['test_methods'],
                'optional': [],
                'defaults': {},
                'types': {'test_methods': list}
            }
        }
    
    def validate_and_resolve(self, tool_name: str, raw_params: Dict) -> ResolvedInput:
        """Validate and resolve tool inputs with improved accuracy."""
        notes = []
        confidence = 1.0
        
        # Check if tool exists
        if tool_name not in self.tool_schemas:
            # Try fuzzy matching
            matched_tool = self._fuzzy_match_tool(tool_name)
            if matched_tool:
                notes.append(f"Corrected tool name: {tool_name} -> {matched_tool}")
                tool_name = matched_tool
                confidence *= 0.9
            else:
                return ResolvedInput(
                    tool_name=tool_name,
                    params=raw_params,
                    confidence=0.0,
                    resolution_notes=[f"Unknown tool: {tool_name}"],
                    is_valid=False
                )
        
        # Get schema
        schema = self.tool_schemas[tool_name]
        resolved_params = {}
        
        # Add required parameters
        for param in schema['required']:
            if param in raw_params:
                resolved_params[param] = self._cast_type(
                    raw_params[param],
                    schema['types'].get(param, str)
                )
            else:
                # Try to infer
                inferred = self._infer_parameter(tool_name, param)
                if inferred is not None:
                    resolved_params[param] = inferred
                    notes.append(f"Inferred {param} from context")
                    confidence *= 0.8
                else:
                    # Use placeholder with low confidence
                    resolved_params[param] = self._get_placeholder(tool_name, param)
                    notes.append(f"Used placeholder for missing {param}")
                    confidence *= 0.3  # Lower confidence for placeholders
        
        # Add optional parameters
        for param in schema['optional']:
            if param in raw_params:
                resolved_params[param] = self._cast_type(
                    raw_params[param],
                    schema['types'].get(param, str)
                )
            else:
                resolved_params[param] = schema['defaults'].get(param)
        
        # Tool-specific resolution with improved accuracy
        resolved_params = self._resolve_tool_specific_improved(tool_name, resolved_params, notes)
        
        return ResolvedInput(
            tool_name=tool_name,
            params=resolved_params,
            confidence=confidence,
            resolution_notes=notes,
            is_valid=confidence > 0.3
        )
    
    def _resolve_tool_specific_improved(self, tool_name: str, params: Dict, notes: List[str]) -> Dict:
        """Improved tool-specific resolution with better file matching."""
        
        if tool_name == 'code_extractor':
            # Improved file path resolution for code extractor
            if not params.get('file_path') and self.indexer:
                element_type = params.get('element_type', 'method')
                element_name = params.get('element_name', '')
                
                if element_type == 'test':
                    # Find test file with improved matching
                    test_path = self.indexer.find_test_path(element_name)
                    if test_path:
                        params['file_path'] = test_path
                        notes.append(f"Resolved test file: {test_path}")
                    else:
                        # Try to find any file containing the test
                        locations = self.indexer.find_method_locations(element_name)
                        test_locations = [loc for loc in locations if 'test' in loc[1].lower()]
                        if test_locations:
                            params['file_path'] = test_locations[0][1]
                            notes.append(f"Found test in: {test_locations[0][1]}")
                
                elif element_type == 'class':
                    # Find class file with verification
                    class_path = self.indexer.find_class_path(element_name)
                    if class_path:
                        # Verify the class actually exists in the file
                        if self._verify_element_in_file(class_path, element_name, 'class'):
                            params['file_path'] = class_path
                            notes.append(f"Verified class file: {class_path}")
                        else:
                            notes.append(f"Class {element_name} not found in {class_path}")
                
                elif element_type == 'method':
                    # Find method with location verification
                    locations = self.indexer.find_method_locations(element_name)
                    if locations:
                        # Prefer non-test files for regular methods
                        non_test_locations = [loc for loc in locations if 'test' not in loc[1].lower()]
                        if non_test_locations:
                            params['file_path'] = non_test_locations[0][1]
                            notes.append(f"Found method in: {non_test_locations[0][1]}")
                        elif locations:
                            params['file_path'] = locations[0][1]
                            notes.append(f"Found method in test file: {locations[0][1]}")
        
        elif tool_name == 'field_dependency_analyzer':
            # Resolve file path with class verification
            if not params.get('file_path') and self.indexer:
                class_path = self.indexer.find_class_path(params.get('class_name', ''))
                if class_path and self._verify_element_in_file(class_path, params.get('class_name', ''), 'class'):
                    params['file_path'] = class_path
                    notes.append(f"Verified class for field analysis: {class_path}")
        
        elif tool_name == 'call_graph_builder':
            # Resolve file path from method with verification
            if not params.get('file_path') and params.get('method_name') and self.indexer:
                locations = self.indexer.find_method_locations(params['method_name'])
                if locations:
                    # Prefer the location from buggy file if available
                    buggy_file = self.bug_info.get('buggy_file_path')
                    if buggy_file:
                        buggy_locations = [loc for loc in locations if buggy_file in loc[1]]
                        if buggy_locations:
                            params['file_path'] = buggy_locations[0][1]
                            notes.append(f"Using method from buggy file: {buggy_locations[0][1]}")
                        else:
                            params['file_path'] = locations[0][1]
                            notes.append(f"Found method in: {locations[0][1]}")
                    else:
                        params['file_path'] = locations[0][1]
                        notes.append(f"Found method in: {locations[0][1]}")
        
        elif tool_name == 'similar_method_search':
            # Ensure method body is not empty
            if not params.get('method_body'):
                if self.bug_info.get('buggy_method'):
                    params['method_body'] = self.bug_info['buggy_method']
                    notes.append("Used buggy method from context")
                else:
                    params['method_body'] = "// No method body available"
                    notes.append("No method body available - using placeholder")
        
        return params
    
    def _verify_element_in_file(self, file_path: str, element_name: str, element_type: str) -> bool:
        """Verify that an element actually exists in the file."""
        if not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if element_type == 'class':
                # Check for class or interface definition in Java
                java_type = re.search(
                    rf'\b(class|interface)\s+{re.escape(element_name)}\b',
                    content
                )
                # Python still only has class
                python_class = re.search(
                    rf'^class\s+{re.escape(element_name)}\b',
                    content,
                    re.MULTILINE
                )
                return java_type is not None or python_class is not None

            elif element_type == 'method':
                # Check for method definition
                java_method = re.search(rf'\b{re.escape(element_name)}\s*\([^)]*\)\s*\{{', content)
                python_method = re.search(rf'^\s*def\s+{re.escape(element_name)}\s*\(', content, re.MULTILINE)
                return java_method is not None or python_method is not None
            
        except Exception:
            return False
        
        return False
    
    def _fuzzy_match_tool(self, tool_name: str) -> Optional[str]:
        """Fuzzy match tool name."""
        tool_names = list(self.tool_schemas.keys())
        # Add common aliases
        tool_aliases = {
            'test_code_extractor': 'code_extractor',
            'test_extractor': 'code_extractor',
            'method_extractor': 'code_extractor',
            'field_analyzer': 'field_dependency_analyzer',
            'call_graph': 'call_graph_builder',
            'api_finder': 'api_usage_finder',
            'coverage': 'coverage_runner'
        }
        
        # Check aliases first
        if tool_name.lower() in tool_aliases:
            return tool_aliases[tool_name.lower()]
        
        # Fuzzy match
        best_match = process.extractOne(tool_name, tool_names, scorer=fuzz.ratio)
        
        if best_match and best_match[1] > 70:
            return best_match[0]
        
        return None
    
    def _infer_parameter(self, tool_name: str, param: str) -> Optional[Any]:
        """Infer parameter from context."""
        if param == 'method_body' and 'buggy_method' in self.bug_info:
            return self.bug_info['buggy_method']
        
        if param == 'element_name':
            if tool_name == 'code_extractor':
                # Try to infer from context
                if 'failing_tests' in self.bug_info and self.bug_info['failing_tests']:
                    return self.bug_info['failing_tests'][0]
                if 'buggy_method_name' in self.bug_info:
                    return self.bug_info['buggy_method_name']
        
        if param == 'class_name':
            if 'buggy_class' in self.bug_info:
                return self.bug_info['buggy_class']
            if 'buggy_file_path' in self.bug_info:
                return self._extract_class_from_path(self.bug_info['buggy_file_path'])
        
        if param == 'method_name':
            if 'buggy_method_name' in self.bug_info:
                return self.bug_info['buggy_method_name']
        
        if param == 'file_path' and 'buggy_file_path' in self.bug_info:
            return self.bug_info['buggy_file_path']
        
        if param == 'test_methods' and 'failing_tests' in self.bug_info:
            return self.bug_info['failing_tests']
        
        return None
    
    def _get_placeholder(self, tool_name: str, param: str) -> Any:
        """Get a reasonable placeholder for missing parameter."""
        placeholders = {
            'method_body': '// Method body not available',
            'element_name': 'unknown_element',
            'class_name': 'UnknownClass',
            'method_name': 'unknownMethod',
            'file_path': '',
            'test_methods': [],
            'api_class': 'Object',
            'api_method': 'toString',
            'top_k': 5,
            'max_depth': 2,
            'direction': 'both',
            'filter_keywords': [],
            'max_examples': 5,
            'element_type': 'method',
            'language': None
        }
        return placeholders.get(param, '')
    
    def _extract_class_from_path(self, file_path: str) -> str:
        """Extract class name from file path."""
        basename = os.path.basename(file_path)
        class_name = basename.rsplit('.', 1)[0]
        return class_name
    
    def _cast_type(self, value: Any, expected_type: type) -> Any:
        """Cast value to expected type."""
        if expected_type == int:
            try:
                return int(value)
            except (ValueError, TypeError):
                return 0
        elif expected_type == list:
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                return [value]
            else:
                return []
        elif expected_type == str:
            return str(value) if value is not None else ''
        else:
            return value
    
    def validate_and_correct(self, tool_name: str, params: Dict) -> ValidationResult:
        """Validate and correct tool inputs (backward compatibility)."""
        result = self.validate_and_resolve(tool_name, params)
        
        return ValidationResult(
            is_valid=result.is_valid,
            corrected_params=result.params,
            errors=[] if result.is_valid else result.resolution_notes,
            warnings=result.resolution_notes if result.is_valid else [],
            confidence=result.confidence
        )