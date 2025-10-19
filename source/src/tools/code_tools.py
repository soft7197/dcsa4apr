# src/tools/code_tools.py
"""
Enhanced and fixed code tools implementation.
Consolidated TestCodeExtractor functionality into CodeExtractor.
All tools are validated and work properly with LLM context manager.
"""

import logging
import os
import re
import tempfile
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple, Any
import ast
import subprocess
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import xml.etree.ElementTree as ET

# Try to import optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using numpy for similarity search")

try:
    import javalang
    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False
    logging.warning("javalang not available, using regex for Java parsing")


class SimilarMethodSearcher:
    """Search for similar methods with caching and optimization."""
    
    def __init__(self, index_path: str = None, metadata_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.index = None
        self.metadata = []
        self.model = None
        self.tokenizer = None
        self._embedding_cache = {}
        
        if index_path and metadata_path:
            self._load_index(index_path, metadata_path)
        
        self._model_loaded = False
    
    def _load_index(self, index_path: str, metadata_path: str):
        """Load pre-built index and metadata."""
        try:
            if FAISS_AVAILABLE and os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                self.logger.info(f"Loaded FAISS index from {index_path}")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                self.logger.info(f"Loaded {len(self.metadata)} methods metadata")
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model_loaded:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.model = AutoModel.from_pretrained("microsoft/codebert-base")
            self._model_loaded = True
            self.logger.info("Loaded CodeBERT model")
        except Exception as e:
            self.logger.error(f"Error loading CodeBERT model: {e}")
            self._model_loaded = False
    
    def search(self, method_body: str, top_k: int = 5, filter_keywords: List[str] = None) -> List[Dict]:
        """Search for similar methods with fallback strategies."""
        if not method_body:
            self.logger.warning("Empty method body provided")
            return []
        
        results = []
        
        # Try FAISS search first if available
        if self.index and FAISS_AVAILABLE:
            embedding = self._get_embedding(method_body)
            if embedding is not None:
                try:
                    distances, indices = self.index.search(
                        embedding.reshape(1, -1), 
                        min(top_k * 2, len(self.metadata))  # Get more for filtering
                    )
                    
                    for dist, idx in zip(distances[0], indices[0]):
                        if idx >= 0 and idx < len(self.metadata):
                            method_info = self.metadata[idx].copy()
                            method_info['similarity'] = float(1 - dist)
                            results.append(method_info)
                    
                except Exception as e:
                    self.logger.error(f"FAISS search error: {e}")
        
        # Fallback to simple text similarity if needed
        if not results and self.metadata:
            results = self._text_similarity_search(method_body, top_k * 2)
        
        # Apply keyword filter if specified
        if filter_keywords:
            results = [r for r in results if any(
                kw.lower() in str(r.get('code', '')).lower() 
                for kw in filter_keywords
            )]
        
        return results[:top_k]
    
    def _get_embedding(self, method_body: str) -> Optional[np.ndarray]:
        """Generate embedding for method body."""
        cache_key = hashlib.md5(method_body.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        self._load_model()
        
        if not self._model_loaded:
            return None
        
        try:
            import torch
            
            inputs = self.tokenizer(
                method_body,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            result = embedding[0] if embedding.shape[0] > 0 else None
            
            if result is not None:
                self._embedding_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return None
    
    def _text_similarity_search(self, method_body: str, top_k: int) -> List[Dict]:
        """Fallback text-based similarity search."""
        from difflib import SequenceMatcher
        
        similarities = []
        for i, method_meta in enumerate(self.metadata):
            if 'code' in method_meta:
                similarity = SequenceMatcher(
                    None, 
                    method_body, 
                    method_meta['code']
                ).ratio()
                
                result = method_meta.copy()
                result['similarity'] = similarity
                similarities.append(result)
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

class CodeExtractor:
    """
    Enhanced code extractor that handles both regular code and test code.
    Replaces the old TestCodeExtractor functionality.
    """
    
    def __init__(self, project_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.project_path = project_path
        self._extraction_cache = {}
        self._file_cache = {}
    
    def extract(self, file_path: str, element_name: str, 
                element_type: str = 'method', language: str = None) -> Dict:
        """
        Main extraction method that handles all element types.
        
        Args:
            file_path: Path to the file (can be relative or absolute)
            element_name: Name of the element to extract
            element_type: Type of element ('method', 'class', 'field', 'test')
            language: Programming language (auto-detected if None)
        
        Returns:
            Dictionary with extracted code and metadata
        """
        # Resolve file path
        resolved_path = self._resolve_file_path(file_path, element_name, element_type)
        if not resolved_path:
            return self._empty_result(element_name, element_type)
        
        # Auto-detect language if not provided
        if not language:
            language = self._detect_language(resolved_path)
        
        # Check cache
        cache_key = f"{resolved_path}:{element_name}:{element_type}"
        if cache_key in self._extraction_cache:
            return self._extraction_cache[cache_key]
        
        # Read file content
        content = self._read_file(resolved_path)
        if not content:
            return self._empty_result(element_name, element_type)
        
        # Extract based on element type
        result = None
        if element_type == 'test':
            result = self._extract_test(content, element_name, language)
        elif element_type == 'method':
            result = self._extract_method(content, element_name, language)
        elif element_type == 'class':
            result = self._extract_class(content, element_name, language)
        elif element_type == 'field':
            result = self._extract_field(content, element_name, language)
        else:
            self.logger.warning(f"Unknown element type: {element_type}")
            result = self._empty_result(element_name, element_type)
        
        # Cache successful extractions
        if result and result.get('code'):
            self._extraction_cache[cache_key] = result
        
        return result
    
    def extract_test_code(self, test_name: str, test_path: str = None) -> str:
        """
        Backward compatibility method for test extraction.
        """
        result = self.extract(
            file_path=test_path or '',
            element_name=test_name,
            element_type='test'
        )
        return result.get('code', '')
    
    def extract_method_body(self, file_path: str, method_name: str, 
                           language: str = None) -> str:
        """
        Backward compatibility method for method extraction.
        """
        result = self.extract(
            file_path=file_path,
            element_name=method_name,
            element_type='method',
            language=language
        )
        return result.get('code', '')
    
    def _resolve_file_path(self, file_path: str, element_name: str, element_type: str) -> Optional[str]:
        """Resolve file path for any element (class, method, field, interface, enum, test, etc.)."""

        candidate_paths = []

        # 1. Direct absolute path
        if file_path and os.path.isabs(file_path) and os.path.exists(file_path):
            return file_path

        # 2. Relative path from project root
        if self.project_path and file_path:
            full_path = os.path.join(self.project_path, file_path)
            if os.path.exists(full_path):
                return full_path
            candidate_paths.append(full_path)

        # 3. Element-specific search
        if self.project_path:
            found_file = self._search_file_by_element(element_name, element_type)
            if found_file and os.path.exists(found_file):
                return found_file
            candidate_paths.append(found_file)

        # 4. Validate candidates
        for path in candidate_paths:
            if path and os.path.exists(path):
                return path

        # 5. Log failure
        if file_path:
            self.logger.warning(
                f"Could not resolve valid file path for {element_type} '{element_name}'. Tried: {candidate_paths}"
            )

        return None
    
    def _find_test_file(self, test_name: str) -> Optional[str]:
        """Find test file by test name."""
        if not self.project_path:
            return None
        
        # Extract class name from test method name
        class_name = test_name.split('::')[0] if '::' in test_name else test_name
        if '.' in class_name:
            class_name = class_name.split('.')[-1]
        
        # Common test directories
        test_dirs = ['src/test/java', 'test', 'tests', 'src/test', 'test/java']
        
        for test_dir in test_dirs:
            test_path = os.path.join(self.project_path, test_dir)
            if os.path.exists(test_path):
                # Search for test file
                for root, dirs, files in os.walk(test_path):
                    for file in files:
                        if class_name in file and file.endswith(('.java', '.py')):
                            return os.path.join(root, file)
        
        return None
        
    def _search_file_by_element(self, element_name: str, element_type: str) -> Optional[str]:
        """Search for file containing the element, with prioritization.
        Supports class, interface, enum, annotation, method, field, test.
        """
        if not self.project_path or not element_name:
            return None

        matches = []
        for root, dirs, files in os.walk(self.project_path):
            # Prevent deep traversal into irrelevant dirs
            depth = root[len(self.project_path):].count(os.sep)
            if depth > 6:
                dirs.clear()
                continue

            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['build', 'target', 'out', 'node_modules']]

            for file in files:
                if not file.endswith(('.java', '.py')):
                    continue

                file_path = os.path.join(root, file)
                content = self._read_file(file_path)

                if not content:
                    continue

                # Match element by type
                if element_type in ['class', 'method', 'field', 'interface', 'enum', 'annotation', 'test']:
                    # Look for exact element definition
                    patterns = [
                        rf'\bclass\s+{element_name}\b',
                        rf'\binterface\s+{element_name}\b',
                        rf'\benum\s+{element_name}\b',
                        rf'@interface\s+{element_name}\b',
                        rf'\b{element_name}\s*\('  # methods
                    ]
                    if any(re.search(p, content) for p in patterns):
                        matches.append(file_path)

                # Fallback: name appears in file
                elif element_name in content:
                    matches.append(file_path)

        if not matches:
            return None

        # Prioritize matches
        matches.sort(key=lambda p: (
            0 if "src/main/java" in p else (1 if "src/test/java" in p else 2),
            0 if os.path.basename(p).startswith(element_name) else 1,
            len(p)
        ))

        return matches[0]
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        if file_path.endswith('.java'):
            return 'java'
        elif file_path.endswith('.py'):
            return 'python'
        else:
            return 'unknown'
    
    def _read_file(self, file_path: str) -> str:
        """Read file with caching."""
        if file_path in self._file_cache:
            return self._file_cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                self._file_cache[file_path] = content
                return content
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def _extract_test(self, content: str, test_name: str, language: str) -> Dict:
        """Extract test method with assertions and setup."""
        if language == 'java':
            return self._extract_java_test(content, test_name)
        elif language == 'python':
            return self._extract_python_test(content, test_name)
        else:
            # Fallback to method extraction
            return self._extract_method(content, test_name, language)
    
    def _extract_java_test(self, content: str, test_name: str) -> Dict:
        """Extract Java test method with annotations and assertions."""
        lines = content.split('\n')
        
        # Handle both simple method name and fully qualified name
        method_name = test_name.split('::')[-1] if '::' in test_name else test_name
        if '.' in method_name:
            method_name = method_name.split('.')[-1]
        
        # Find test method
        method_start = -1
        for i, line in enumerate(lines):
            # Look for @Test annotation or test method signature
            if method_name in line and ('void' in line or '@Test' in lines[max(0, i-5):i]):
                method_start = i
                # Include annotations
                while i > 0 and lines[i-1].strip().startswith('@'):
                    i -= 1
                    method_start = i
                break
        
        if method_start == -1:
            return self._empty_result(test_name, 'test')
        
        # Extract method with proper brace matching
        code, method_end = self._extract_balanced_braces(lines, method_start)
        
        # Extract setup methods if present
        setup_code = self._extract_setup_methods(lines, method_start)
        
        return {
            'code': code,
            'type': 'test',
            'name': test_name,
            'setup': setup_code,
            'language': 'java',
            'lines': (method_start + 1, method_end + 1)
        }
    
    def _extract_python_test(self, content: str, test_name: str) -> Dict:
        """Extract Python test method."""
        lines = content.split('\n')
        
        # Find test method
        pattern = rf'^\s*def\s+{re.escape(test_name)}\s*\('
        method_start = -1
        
        for i, line in enumerate(lines):
            if re.match(pattern, line):
                method_start = i
                break
        
        if method_start == -1:
            return self._empty_result(test_name, 'test')
        
        # Extract method by indentation
        initial_indent = len(lines[method_start]) - len(lines[method_start].lstrip())
        method_end = method_start + 1
        
        for i in range(method_start + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= initial_indent:
                    method_end = i
                    break
        else:
            method_end = len(lines)
        
        code = '\n'.join(lines[method_start:method_end])
        assertions = self._extract_assertions(code)
        
        return {
            'code': code,
            'type': 'test',
            'name': test_name,
            'assertions': assertions,
            'language': 'python',
            'lines': (method_start + 1, method_end)
        }
    
    def _extract_method(self, content: str, method_name: str, language: str) -> Dict:
        """Extract regular method."""
        if language == 'java':
            code = self._extract_java_method(content, method_name)
        elif language == 'python':
            code = self._extract_python_method(content, method_name)
        else:
            code = ""
        
        return {
            'code': code,
            'type': 'method',
            'name': method_name,
            'language': language
        }
    
    def _extract_java_method(self, content: str, method_name: str) -> str:
        """Extract Java method with multiple strategies."""
        # Try AST parsing if available
        if JAVALANG_AVAILABLE:
            result = self._extract_java_method_ast(content, method_name)
            if result:
                return result
        
        # Fallback to regex
        return self._extract_java_method_regex(content, method_name)
    
    def _extract_java_method_ast(self, content: str, method_name: str) -> str:
        """Extract Java method using AST."""
        if not JAVALANG_AVAILABLE:
            return ""
        
        try:
            tree = javalang.parse.parse(content)
            found_codes = ""
            for _, node in tree.filter(javalang.tree.MethodDeclaration):
                if node.name == method_name:
                    # Get method position and extract
                    if hasattr(node, 'position') and node.position:
                        lines = content.split('\n')
                        start_line = node.position.line - 1
                        
                        # Find end by brace matching
                        code, _ = self._extract_balanced_braces(lines, start_line)
                        found_codes += code + "\n"
            return found_codes.strip()
        except Exception as e:
            self.logger.debug(f"AST parsing failed: {e}")
        
        return ""
    
    def _extract_java_method_regex(self, content: str, method_name: str) -> str:
        """Extract Java method using regex."""
        lines = content.split('\n')
        
        # Patterns for method signatures
        patterns = [
            rf'public\s+\w+\s+{re.escape(method_name)}\s*\(',
            rf'private\s+\w+\s+{re.escape(method_name)}\s*\(',
            rf'protected\s+\w+\s+{re.escape(method_name)}\s*\(',
            rf'static\s+\w+\s+{re.escape(method_name)}\s*\(',
            rf'void\s+{re.escape(method_name)}\s*\(',
            rf'\w+\s+{re.escape(method_name)}\s*\('
        ]
        found_codes = ""
        for pattern in patterns:
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    code, _ = self._extract_balanced_braces(lines, i)
                    if code:
                        found_codes += code + "\n"
        if found_codes:
            return found_codes.strip()
        
        return ""
    
    def _extract_python_method(self, content: str, method_name: str) -> str:
        """Extract Python method."""
        lines = content.split('\n')
        
        pattern = rf'^\s*def\s+{re.escape(method_name)}\s*\('
        for i, line in enumerate(lines):
            if re.match(pattern, line):
                # Extract by indentation
                initial_indent = len(line) - len(line.lstrip())
                end_idx = i + 1
                
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        line_indent = len(lines[j]) - len(lines[j].lstrip())
                        if line_indent <= initial_indent:
                            end_idx = j
                            break
                else:
                    end_idx = len(lines)
                
                return '\n'.join(lines[i:end_idx])
        
        return ""
    
    def _extract_class(self, content: str, class_name: str, language: str) -> Dict:
        """Extract class definition."""
        if language == 'java':
            code = self._extract_java_class(content, class_name)
        elif language == 'python':
            code = self._extract_python_class(content, class_name)
        else:
            code = ""
        
        return {
            'code': code,
            'type': 'class',
            'name': class_name,
            'language': language
        }
    def _extract_java_class(self, content: str, class_name: str) -> str:
        """Extract Java class, interface, or enum."""
        lines = content.split('\n')
        # Match class/interface/enum declaration with optional modifiers
        pattern = rf'(public\s+|protected\s+|private\s+|abstract\s+|final\s+)?(class|interface|enum)\s+{re.escape(class_name)}\b'
        
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                code, _ = self._extract_balanced_braces(lines, i)
                return code
        
        return ""
    
    def _extract_python_class(self, content: str, class_name: str) -> str:
        """Extract Python class."""
        lines = content.split('\n')
        
        pattern = rf'^class\s+{re.escape(class_name)}'
        for i, line in enumerate(lines):
            if re.match(pattern, line):
                # Extract by indentation
                end_idx = i + 1
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and not lines[j].startswith(' '):
                        end_idx = j
                        break
                else:
                    end_idx = len(lines)
                
                return '\n'.join(lines[i:end_idx])
        
        return ""
    
    def _extract_field(self, content: str, field_name: str, language: str) -> Dict:
        """Extract field declaration and usages."""
        if language == 'java':
            declaration = self._extract_java_field(content, field_name)
        elif language == 'python':
            declaration = self._extract_python_field(content, field_name)
        else:
            declaration = ""
        
        # Find usages
        usages = self._find_field_usages(content, field_name)
        
        return {
            'code': declaration,
            'type': 'field',
            'name': field_name,
            'language': language,
            'usages': usages
        }
    
    def _extract_java_field(self, content: str, field_name: str) -> str:
        """Extract Java field declaration."""
        pattern = rf'(private|public|protected|static|final|\s)+\w+\s+{re.escape(field_name)}\s*[;=]'
        match = re.search(pattern, content)
        if match:
            # Extract the full line
            lines = content.split('\n')
            for line in lines:
                if field_name in line and re.search(pattern, line):
                    return line.strip()
        return ""
    
    def _extract_python_field(self, content: str, field_name: str) -> str:
        """Extract Python field (class or instance variable)."""
        patterns = [
            rf'self\.{re.escape(field_name)}\s*=',  # Instance variable
            rf'^{re.escape(field_name)}\s*=',  # Class variable
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                # Get the full line
                start = content.rfind('\n', 0, match.start()) + 1
                end = content.find('\n', match.end())
                if end == -1:
                    end = len(content)
                return content[start:end].strip()
        
        return ""
    
    def _find_field_usages(self, content: str, field_name: str) -> List[str]:
        """Find all usages of a field."""
        usages = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if field_name in line:
                usages.append(f"Line {i+1}: {line.strip()[:100]}")
        
        return usages[:10]  # Limit to 10 usages
    
    def _extract_balanced_braces(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Extract code with balanced braces."""
        brace_count = 0
        in_method = False
        end_idx = start_idx
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            
            # Count braces
            open_braces = line.count('{')
            close_braces = line.count('}')
            
            brace_count += open_braces
            if open_braces > 0:
                in_method = True
            
            brace_count -= close_braces
            
            if in_method and brace_count == 0:
                end_idx = i
                break
        
        return '\n'.join(lines[start_idx:end_idx + 1]), end_idx
    
    def _extract_assertions(self, code: str) -> List[Dict]:
        """Extract assertion statements from test code."""
        assertions = []
        
        # Java assertions
        java_patterns = [
            r'assert(?:Equals|True|False|Null|NotNull|Same|NotSame|That|Throws)\s*\([^)]+\)',
            r'assertEquals\s*\([^)]+\)',
            r'assertTrue\s*\([^)]+\)',
            r'assertFalse\s*\([^)]+\)',
            r'fail\s*\([^)]*\)'
        ]
        
        # Python assertions
        python_patterns = [
            r'assert\s+[^,\n]+',
            r'self\.assert(?:Equal|True|False|In|NotIn|Is|IsNot|IsNone|IsNotNone)\s*\([^)]+\)',
            r'pytest\.raises\s*\([^)]+\)'
        ]
        
        all_patterns = java_patterns + python_patterns
        
        for pattern in all_patterns:
            for match in re.finditer(pattern, code):
                assertion_text = match.group(0)
                line_num = code[:match.start()].count('\n') + 1
                
                assertions.append({
                    'text': assertion_text,
                    'line': line_num,
                    'type': self._classify_assertion(assertion_text)
                })
        
        return assertions
    
    def _classify_assertion(self, assertion_text: str) -> str:
        """Classify the type of assertion."""
        if 'Equal' in assertion_text:
            return 'equality'
        elif 'True' in assertion_text:
            return 'boolean_true'
        elif 'False' in assertion_text:
            return 'boolean_false'
        elif 'Null' in assertion_text:
            return 'null_check'
        elif 'fail' in assertion_text:
            return 'explicit_fail'
        else:
            return 'other'
    
    def _extract_setup_methods(self, lines: List[str], test_start: int) -> str:
        """Extract @Before or setUp methods for context."""
        setup_code = []
        
        # Look for @Before or @BeforeEach methods (Java)
        for i in range(max(0, test_start - 50), test_start):
            if '@Before' in lines[i] or '@BeforeEach' in lines[i]:
                # Extract the setup method
                for j in range(i, min(i + 30, test_start)):
                    if '{' in lines[j]:
                        code, end = self._extract_balanced_braces(lines, j)
                        setup_code.append(code)
                        break
        
        return '\n'.join(setup_code) if setup_code else ""
    
    def _empty_result(self, element_name: str, element_type: str) -> Dict:
        """Return empty result structure."""
        return {
            'code': '',
            'type': element_type,
            'name': element_name,
            'error': f"Could not extract {element_type}: {element_name}"
        }

class CallGraphBuilder:
    """Build call graphs with caching and optimization."""
    
    def __init__(self, project_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.project_path = project_path
        self._call_graph_cache = {}
        self.code_extractor = CodeExtractor(project_path)
    
    def build_call_graph(self, method_name: str, file_path: str = None,
                        direction: str = 'both', max_depth: int = 2) -> Dict:
        """
        Build call graph for a method.
        
        Args:
            method_name: Name of the method
            file_path: Optional file path
            direction: 'callers', 'callees', or 'both'
            max_depth: Maximum depth to traverse
        
        Returns:
            Dictionary with callers and/or callees
        """
        cache_key = f"{file_path}:{method_name}:{direction}:{max_depth}"
        if cache_key in self._call_graph_cache:
            return self._call_graph_cache[cache_key]
        
        result = {
            'method': method_name,
            'file': file_path
        }
        
        if direction in ['callers', 'both']:
            result['callers'] = self._find_callers(method_name, file_path, max_depth)
        
        if direction in ['callees', 'both']:
            result['callees'] = self._find_callees(method_name, file_path, max_depth)
        
        self._call_graph_cache[cache_key] = result
        return result
    
    def _find_callers(self, method_name: str, file_path: str, max_depth: int) -> List[Dict]:
        """Find methods that call the target method within the project."""
        callers = []

        if not self.project_path:
            return callers

        # Regex for method invocation (avoiding keywords/annotations)
        search_pattern = rf'(?<!@)\b{re.escape(method_name)}\s*\('

        for root, dirs, files in os.walk(self.project_path):
            # Limit search depth
            depth = root[len(self.project_path):].count(os.sep)
            if depth > 5:
                continue

            # Skip build/system dirs
            dirs[:] = [d for d in dirs if d not in ['.git', 'build', 'target', 'node_modules']]

            for file in files:
                if file.endswith('.java'):  # stick to Java for consistency
                    current_path = os.path.join(root, file)

                    try:
                        with open(current_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        if re.search(search_pattern, content):
                            caller_methods = self._extract_calling_methods(
                                content, method_name, current_path
                            )
                            for cm in caller_methods:
                                if cm not in callers:  # deduplicate
                                    callers.append(cm)

                    except Exception as e:
                        self.logger.debug(f"Error processing {current_path}: {e}")

            if len(callers) >= 20:  # global limit
                break

        # Optional recursion if max_depth > 1
        if max_depth > 1:
            expanded = []
            for caller in callers:
                try:
                    deeper_callers = self._find_callers(
                        caller['method'], caller.get('file', ''), max_depth - 1
                    )
                    caller['callers'] = deeper_callers
                except Exception:
                    caller['callers'] = []
                expanded.append(caller)
            callers = expanded

        return callers[:20]

    def _find_callees(self, method_name: str, file_path: str, max_depth: int) -> List[Dict]:
        """Find methods called by the target method, optionally with recursion (max_depth)."""
        if file_path:
            method_result = self.code_extractor.extract(file_path, method_name, 'method')
            method_body = method_result.get('code', '')
        else:
            method_body = ""

        if not method_body:
            return []

        callees = []
        seen = set()

        # Regex patterns for different call types
        call_patterns = [
            (r'(\w+)\s*\([^)]*\)', "direct_call"),             # foo(arg)
            (r'(\w+)\s*<[^>]+>\s*\([^)]*\)', "generic_call"),  # foo<T>(arg)
            (r'new\s+(\w+)\s*\([^)]*\)', "constructor"),       # new Class(arg)
            (r'super\.(\w+)\s*\([^)]*\)', "super_call"),       # super.foo(arg)
            (r'this\.(\w+)\s*\([^)]*\)', "this_call"),         # this.foo(arg)
            (r'(\w+)\.(\w+)\s*\([^)]*\)', "qualified_call"),   # ClassName.method(arg) / obj.method(arg)
        ]

        # Keywords to ignore
        ignore_set = {'if', 'while', 'for', 'switch', 'return',
                    'catch', 'try', 'else', 'throw', 'new', 'case'}

        for pattern, call_type in call_patterns:
            for match in re.finditer(pattern, method_body):
                # Handle qualified calls specially
                if call_type == "qualified_call":
                    qualifier, called_method = match.group(1), match.group(2)
                else:
                    qualifier, called_method = None, match.group(1)

                if called_method in ignore_set:
                    continue

                key = f"{qualifier}.{called_method}" if qualifier else called_method
                if key not in seen:
                    seen.add(key)
                    callees.append({
                        'method': called_method,
                        'qualifier': qualifier,
                        'context': match.group(0)[:120],  # capture more context
                        'type': call_type
                    })

        # If recursion requested (max_depth > 1), expand callees
        if max_depth > 1:
            expanded = []
            for callee in callees:
                try:
                    sub_callees = self._find_callees(
                        callee['method'], file_path, max_depth - 1
                    )
                    callee['callees'] = sub_callees
                except Exception:
                    callee['callees'] = []
                expanded.append(callee)
            callees = expanded

        return callees[:15]

    
    def _extract_calling_methods(self, content: str, called_method: str, 
                                 file_path: str) -> List[Dict]:
        """Extract methods that contain calls to the target method."""
        calling_methods = []
        
        # Find all methods in the file
        method_pattern = r'(public|private|protected|static|\s)+\w+\s+(\w+)\s*\([^)]*\)\s*\{'
        
        for match in re.finditer(method_pattern, content):
            method_name = match.group(2)
            method_start = match.start()
            
            # Extract method body
            lines = content.split('\n')
            start_line = content[:method_start].count('\n')
            
            method_code, _ = self.code_extractor._extract_balanced_braces(
                lines, start_line
            )
            
            # Check if this method calls our target
            if re.search(rf'\b{re.escape(called_method)}\s*\(', method_code):
                calling_methods.append({
                    'method': method_name,
                    'file': file_path,
                    'type': 'caller'
                })
        
        return calling_methods


class FieldDependencyAnalyzer:
    """Analyze field dependencies in Java classes with optimization."""

    def __init__(self, project_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.project_path = project_path
        self._field_cache = {}
        self.code_extractor = CodeExtractor(project_path)

    def analyze_fields(self, class_name: str, method_name: str,
                       file_path: str = None) -> List[Dict]:
        """
        Analyze which fields are used by a method.

        Args:
            class_name: Name of the class
            method_name: Name of the method
            file_path: Optional file path

        Returns:
            List of field dependencies with usage information
        """
        cache_key = f"{class_name}:{method_name}"
        if cache_key in self._field_cache:
            return self._field_cache[cache_key]

        if not file_path:
            self.logger.debug("No file_path provided to analyze_fields")
            return []

        # Extract method body
        method_result = self.code_extractor.extract(file_path, method_name, 'method')
        method_body = method_result.get('code', '')

        if not method_body:
            file_path = self.code_extractor._resolve_file_path("", method_name, 'method') 
            method_result = self.code_extractor.extract(file_path, method_name, 'method')
            method_body = method_result.get('code', '')

        if not method_body:
            return []
        
        class_name = file_path.split('/')[-1].replace('.java', '')
        # Find class fields
        fields = self._extract_class_fields(class_name, file_path)

        # Analyze field usage in method
        dependencies = []
        for field in fields:
            usage = self._analyze_field_usage(method_body, field['name'])

            if usage['used']:
                dependencies.append({
                    'field': field['name'],
                    'type': field['type'],
                    'modifiers': field['modifiers'],
                    'access_summary': usage['summary'],
                    'usage_count': usage['count'],
                    'initialization': field.get('initialization', 'unknown')
                })

        self._field_cache[cache_key] = dependencies
        return dependencies

    def _extract_class_fields(self, class_name: str, file_path: str) -> List[Dict]:
        """Extract all fields from a class, including generics, arrays, and packages."""
        class_result = self.code_extractor.extract(file_path, class_name, 'class')
        class_body = class_result.get('code', '')

        if not class_body:
            return []

        fields = []

        # Pattern for Java field declarations
        java_pattern = re.compile(
            r'(?:@[A-Za-z0-9_]+\s*)*'                             # annotations
            r'(?P<modifiers>(?:public|protected|private|static|final|'
            r'transient|volatile|synchronized|native|strictfp|\s)+)'
            r'(?P<type>[\w<>\[\].?]+)\s+'                         # type (package + generics + arrays)
            r'(?P<name>\w+)\s*'
            r'(?:=\s*[^;]+)?;'                                    # optional initialization
        )

        for match in java_pattern.finditer(class_body):
            field_type = match.group("type")
            field_name = match.group("name")

            # Skip constructs that look like methods
            if field_type in ['void', 'return', 'class', 'interface']:
                continue

            modifiers = match.group("modifiers").strip().split()
            initialization = "=" in match.group(0)

            fields.append({
                'name': field_name,
                'type': field_type,
                'modifiers': modifiers,
                'initialization': initialization
            })

        return fields

    def _analyze_field_usage(self, method_body: str, field_name: str) -> Dict:
        """Analyze how a field is used in a method (read/write/increment)."""
        usage = {
            'used': False,
            'summary': 'none',
            'count': 0
        }

        if field_name not in method_body:
            return usage

        # Patterns for different types of field access (this. or bare)
        read_pattern = re.compile(rf'(\bthis\.{re.escape(field_name)}\b|\b{re.escape(field_name)}\b)(?!\s*=)')
        write_pattern = re.compile(rf'(\bthis\.{re.escape(field_name)}\b|\b{re.escape(field_name)}\b)\s*=')
        increment_pattern = re.compile(rf'(\bthis\.{re.escape(field_name)}\b|\b{re.escape(field_name)}\b)\s*(\+\+|--|\+=|-=)')

        has_read = bool(read_pattern.search(method_body))
        has_write = bool(write_pattern.search(method_body) or increment_pattern.search(method_body))

        # Count total occurrences
        total_matches = (
            len(read_pattern.findall(method_body)) +
            len(write_pattern.findall(method_body)) +
            len(increment_pattern.findall(method_body))
        )

        usage['count'] = total_matches
        usage['used'] = total_matches > 0

        if has_read and has_write:
            usage['summary'] = 'read_write'
        elif has_write:
            usage['summary'] = 'write'
        elif has_read:
            usage['summary'] = 'read'

        return usage
    


class APIUsageFinder:
    """Find usage examples of APIs in the codebase."""

    def __init__(self, project_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.project_path = project_path
        self._usage_cache: Dict[str, List[Dict]] = {}

    def find_usage(self, api_class: str, api_method: str,
                   max_examples: int = 5) -> List[Dict]:
        """
        Find usage examples of an API method.

        Args:
            api_class: API class name
            api_method: API method name
            max_examples: Maximum number of examples to return

        Returns:
            List of usage examples with context
        """
        cache_key = f"{api_class}:{api_method}"
        if cache_key in self._usage_cache:
            return self._usage_cache[cache_key][:max_examples]

        if not self.project_path or not os.path.exists(self.project_path):
            self.logger.warning("Project path not set or does not exist")
            return []

        usages = []

        # Search regex patterns
        patterns = [
            rf'{re.escape(api_class)}\s*\.\s*{re.escape(api_method)}\s*\(',      # Class.method(
            rf'new\s+{re.escape(api_class)}\s*\([^)]*\)\s*\.\s*{re.escape(api_method)}\s*\(',  # new Class().method(
            rf'\b{re.escape(api_method)}\s*\('                                   # method(
        ]

        for root, dirs, files in os.walk(self.project_path):
            # Skip common junk dirs
            dirs[:] = [d for d in dirs if d not in ['.git', 'build', 'target', '__pycache__']]

            for file in files:
                if not file.endswith(('.java', '.py')):
                    continue

                file_path = os.path.join(root, file)

                try:
                    # Skip very large files (>2 MB)
                    if os.path.getsize(file_path) > 2 * 1024 * 1024:
                        continue

                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    for pattern in patterns:
                        for match in re.finditer(pattern, content):
                            context = self._extract_usage_context(content, match.start())

                            usages.append({
                                'file': file_path,
                                'api_class': api_class,
                                'api_method': api_method,
                                'usage': match.group(0),
                                'context': context,
                                'line': content[:match.start()].count('\n') + 1
                            })

                            if len(usages) >= max_examples * 2:
                                break
                        if len(usages) >= max_examples * 2:
                            break

                except Exception as e:
                    self.logger.debug(f"Error reading {file_path}: {e}")

            if len(usages) >= max_examples * 2:
                break

        # Cache results
        self._usage_cache[cache_key] = usages

        # Deduplicate based on context
        seen = set()
        unique_usages = []
        for usage in usages:
            usage_key = (usage['file'], usage['line'])
            if usage_key not in seen:
                seen.add(usage_key)
                unique_usages.append(usage)
                if len(unique_usages) >= max_examples:
                    break

        return unique_usages

    def _extract_usage_context(self, content: str, start: int) -> str:
        """Extract surrounding lines for context."""
        lines = content.split('\n')
        usage_line = content[:start].count('\n')

        # Get ±2 lines around the usage
        context_start = max(0, usage_line - 2)
        context_end = min(len(lines), usage_line + 3)

        context_lines = lines[context_start:context_end]
        return "\n".join(l.strip() for l in context_lines if l.strip())


class CoverageRunner:
    """Run coverage analysis for Defects4J (Java, Cobertura) and BugsInPy (Python)."""

    def __init__(self, project_path: str, language: str = "java"):
        self.project_path = project_path
        self.language = language.lower()
        self.logger = logging.getLogger(__name__)
        self._coverage_cache: Dict[str, List[Dict]] = {}
        self.extractor = CodeExtractor(project_path)

    def get_test_coverage(self, test_methods: List[str]) -> Dict[str, List[Dict]]:
        """Get coverage information for test methods."""
        coverage_data = {}

        for test in test_methods:
            if test in self._coverage_cache:
                coverage_data[test] = self._coverage_cache[test]
                continue

            try:
                if self.language == "java":
                    coverage = self._run_java_coverage(test)
                elif self.language == "python":
                    coverage = self._run_python_coverage(test)
                else:
                    raise ValueError(f"Unsupported language: {self.language}")

                coverage_data[test] = coverage
                self._coverage_cache[test] = coverage
            except Exception as e:
                self.logger.error(f"Error getting coverage for {test}: {e}")
                coverage_data[test] = []

        return coverage_data

    def _run_java_coverage(self, test_method: str) -> List[Dict]:
        """
        Run coverage using Defects4J (Cobertura).
        Returns list of dicts: [{class, filename, methods, lines}]
        where `lines` contains actual code strings instead of just numbers.
        """
        self.logger.info(f"Running Defects4J coverage for test: {test_method}")
        try:
            subprocess.run(
                ["defects4j", "coverage", "-t", test_method],
                cwd=self.project_path,
                check=True,
                capture_output=True,
                text=True
            )

            coverage_file = os.path.join(self.project_path, "coverage.xml")
            if not os.path.exists(coverage_file):
                raise FileNotFoundError("coverage.xml not found after running Defects4J")

            tree = ET.parse(coverage_file)
            root = tree.getroot()

            coverage_data = []
            for pkg in root.findall(".//package"):
                for cls in pkg.findall("classes/class"):
                    class_name = cls.get("name")
                    filename = cls.get("filename")

                    resolved_path = self.extractor._resolve_file_path(filename, filename.split('/')[-1].split('.')[0], 'class')
                    # Read source file once
                    file_path = os.path.join(self.project_path, resolved_path)
                    lines_content = []
                    if os.path.exists(file_path):
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            lines_content = f.readlines()

                    # methods covered
                    methods = []
                    for m in cls.findall("methods/method"):
                        covered_lines = [
                            {
                                "line_number": int(l.get("number")),
                                "code": lines_content[int(l.get("number")) - 1].rstrip()
                            }
                            for l in m.findall("lines/line")
                            if int(l.get("hits", "0")) > 0 and int(l.get("number")) <= len(lines_content)
                        ]
                        methods.append({
                            "name": m.get("name"),
                            "signature": m.get("signature"),
                            "lines": covered_lines[:3]
                        })

                    # lines covered directly at class level
                    covered_lines = [
                        {
                            "line_number": int(l.get("number")),
                            "code": lines_content[int(l.get("number")) - 1].rstrip()
                        }
                        for l in cls.findall("lines/line")
                        if int(l.get("hits", "0")) > 0 and int(l.get("number")) <= len(lines_content)
                    ]

                    coverage_data.append({
                        "class": class_name,
                        "filename": filename,
                        "methods": methods,
                        "lines": covered_lines[:10]
                    })

            return coverage_data

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Defects4J coverage failed: {e.stderr}")
            return []


    def _run_python_coverage(self, test_method: str) -> List[Dict]:
        """
        Run coverage using BugsInPy + coverage.py.
        Returns list of dicts: [{module, functions, lines}]
        where `lines` contains actual code strings instead of just numbers.
        """
        self.logger.info(f"Running BugsInPy coverage for test: {test_method}")
        tmp_json = os.path.join(tempfile.gettempdir(), "coverage.json")

        try:
            subprocess.run(["coverage", "erase"], cwd=self.project_path, check=True)
            subprocess.run(
                ["coverage", "run", "-m", "pytest", test_method],
                cwd=self.project_path,
                check=True,
                capture_output=True,
                text=True
            )
            subprocess.run(["coverage", "json", "-o", tmp_json], cwd=self.project_path, check=True)

            if not os.path.exists(tmp_json):
                raise FileNotFoundError("coverage.json not found after running coverage")

            with open(tmp_json, "r") as f:
                data = json.load(f)

            coverage_data = []
            for file, details in data.get("files", {}).items():
                executed_lines = details.get("executed_lines", [])
                if executed_lines:
                    file_path = os.path.join(self.project_path, file)
                    lines_content = []
                    if os.path.exists(file_path):
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as fsrc:
                            lines_content = fsrc.readlines()

                    covered_lines = [
                        {
                            "line_number": ln,
                            "code": lines_content[ln - 1].rstrip()
                        }
                        for ln in executed_lines
                        if ln <= len(lines_content)
                    ]

                    coverage_data.append({
                        "module": file,
                        "functions": [],
                        "lines": covered_lines
                    })

            return coverage_data

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Coverage.py failed: {e.stderr}")
            return []
