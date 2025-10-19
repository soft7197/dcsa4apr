# src/preprocessing/vector_db_builder.py
import os
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import javalang
import ast
from dataclasses import dataclass

@dataclass
class MethodInfo:
    file_path: str
    class_name: str
    method_name: str
    method_body: str
    start_line: int
    end_line: int
    language: str
    embedding: np.ndarray = None

class VectorDBBuilder:
    def __init__(self, model_name="microsoft/codebert-base", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.methods_data = []
        self.index = None
        
    def extract_java_methods(self, file_path: str) -> List[MethodInfo]:
        """Extract all methods from a Java file."""
        methods = []
        with open(file_path, 'r') as f:
            content = f.read()
            
        try:
            tree = javalang.parse.parse(content)
            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                # Extract method body
                method_start = node.position.line - 1
                method_body = self._extract_method_body(content, node)
                
                class_name = ""
                for p in path:
                    if isinstance(p, javalang.tree.ClassDeclaration):
                        class_name = p.name
                        break
                
                methods.append(MethodInfo(
                    file_path=file_path,
                    class_name=class_name,
                    method_name=node.name,
                    method_body=method_body,
                    start_line=method_start,
                    end_line=method_start + len(method_body.split('\n')),
                    language='java'
                ))
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
        return methods
    
    def extract_python_methods(self, file_path: str) -> List[MethodInfo]:
        """Extract all methods from a Python file."""
        methods = []
        with open(file_path, 'r') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    method_body = ast.get_source_segment(content, node)
                    if method_body and len(method_body) > self.max_length:
                        method_body = method_body[:self.max_length]
                    
                    methods.append(MethodInfo(
                        file_path=file_path,
                        class_name=self._get_class_name(tree, node),
                        method_name=node.name,
                        method_body=method_body,
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        language='python'
                    ))
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
        return methods
    
    def generate_embeddings(self, methods: List[MethodInfo]) -> np.ndarray:
        """Generate CodeBERT embeddings for methods."""
        embeddings = []
        
        for method in methods:
            # Tokenize and truncate if necessary
            inputs = self.tokenizer(
                method.method_body,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                method.embedding = embedding
                embeddings.append(embedding)
        
        return np.vstack(embeddings)
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for similarity search."""
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
    
    def save_index(self, output_dir: str):
        """Save FAISS index and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(output_dir, "methods.index"))
        
        # Save metadata
        with open(os.path.join(output_dir, "methods_metadata.pkl"), 'wb') as f:
            pickle.dump(self.methods_data, f)
    
    def process_project(self, project_path: str, output_dir: str):
        """Process entire project and build vector DB."""
        all_methods = []
        
        # Walk through project directory
        for root, dirs, files in os.walk(project_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith('.java'):
                    methods = self.extract_java_methods(file_path)
                    all_methods.extend(methods)
                elif file.endswith('.py'):
                    methods = self.extract_python_methods(file_path)
                    all_methods.extend(methods)
        
        print(f"Extracted {len(all_methods)} methods")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(all_methods)
        
        # Build FAISS index
        self.build_faiss_index(embeddings)
        
        # Store methods data
        self.methods_data = all_methods
        
        # Save everything
        self.save_index(output_dir)
        
        print(f"Vector DB saved to {output_dir}")