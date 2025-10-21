"""
Enhanced Dataset Loader and Repository Manager for SWE-bench Lite
Handles dataset loading, repository cloning, and environment setup
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import git
from dataclasses import asdict

class SWEBenchLoader:
    """Load and manage SWE-bench Lite dataset"""
    
    def __init__(self, dataset_path: str = "./datasets/swebench_lite.json"):
        self.dataset_path = Path(dataset_path)
        self.workspace_dir = Path("./workspace")
        self.workspace_dir.mkdir(exist_ok=True)
        
    def load_dataset(self) -> List[Dict]:
        """Load SWE-bench Lite dataset"""
        if not self.dataset_path.exists():
            print("Dataset not found. Downloading...")
            self._download_dataset()
        
        with open(self.dataset_path) as f:
            return json.load(f)
    
    def _download_dataset(self):
        """Download SWE-bench Lite from Hugging Face"""
        try:
            from datasets import load_dataset
            
            print("Loading SWE-bench Lite from Hugging Face...")
            dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
            
            data = []
            for item in dataset:
                data.append({
                    "instance_id": item["instance_id"],
                    "repo": item["repo"],
                    "base_commit": item["base_commit"],
                    "problem_statement": item["problem_statement"],
                    "hints_text": item.get("hints_text", ""),
                    "test_patch": item["test_patch"],
                    "patch": item["patch"],
                    "version": item["version"],
                    "FAIL_TO_PASS": item.get("FAIL_TO_PASS", []),
                    "PASS_TO_PASS": item.get("PASS_TO_PASS", [])
                })
            
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dataset_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Downloaded {len(data)} instances")
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("Please download manually from: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite")


class RepositoryManager:
    """Manage git repositories for testing"""
    
    def __init__(self, workspace_dir: str = "./workspace", skip_on_error: bool = True):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        self.skip_on_error = skip_on_error
        
    def setup_repository(self, bug_instance: Dict) -> Path:
        """Clone repository for code extraction only (no testing needed)"""
        
        instance_id = bug_instance["instance_id"]
        repo_name = bug_instance["repo"]
        base_commit = bug_instance["base_commit"]
        
        # Create instance workspace
        instance_dir = self.workspace_dir / instance_id
        repo_dir = instance_dir / "repo"
        
        try:
            # Clone if not exists
            if not repo_dir.exists():
                print(f"Cloning repository: {repo_name}")
                self._clone_repo(repo_name, repo_dir, base_commit)
            else:
                print(f"Repository already exists, reusing: {repo_name}")
            
            # SKIP: Test patch application (not needed - SWE-bench applies it)
            # SKIP: Package installation (not needed - just extracting code)
            
            print(f"  ✅ Repository ready for code extraction")
            return repo_dir
            
        except Exception as e:
            if self.skip_on_error:
                print(f"  ⚠️  Setup warning: {e}")
                print(f"  Continuing anyway - only need code files for extraction")
                # Return the repo_dir anyway if it exists
                if repo_dir.exists():
                    return repo_dir
            raise
    
    def _clone_repo(self, repo_name: str, target_dir: Path, commit: str):
        """Clone repository and checkout specific commit"""
        try:
            # Convert repo name to GitHub URL
            github_url = f"https://github.com/{repo_name}.git"
            
            print(f"  Cloning from {github_url}...")
            
            # Try shallow clone first (faster)
            try:
                repo = git.Repo.clone_from(
                    github_url, 
                    target_dir, 
                    no_checkout=True,
                    depth=1
                )
            except git.GitCommandError:
                # If shallow clone fails, try full clone
                print(f"  Shallow clone failed, trying full clone...")
                repo = git.Repo.clone_from(github_url, target_dir, no_checkout=True)
            
            # Fetch the specific commit
            print(f"  Fetching commit {commit[:8]}...")
            try:
                repo.git.fetch('origin', commit, depth=1)
            except git.GitCommandError:
                # If specific commit fetch fails, fetch more history
                repo.git.fetch('--unshallow')
            
            # Checkout specific commit
            print(f"  Checking out commit {commit[:8]}...")
            repo.git.checkout(commit, force=True)
            
            print(f"  ✅ Repository ready")
            
        except git.GitCommandError as e:
            print(f"  ❌ Git error: {e}")
            # Try alternative: clone with subprocess
            print(f"  Trying alternative clone method...")
            try:
                subprocess.run(
                    f"git clone --depth=1 {github_url} {target_dir}",
                    shell=True,
                    check=True,
                    capture_output=True
                )
                subprocess.run(
                    f"cd {target_dir} && git fetch origin {commit} && git checkout {commit}",
                    shell=True,
                    check=True,
                    capture_output=True
                )
                print(f"  ✅ Repository ready (alternative method)")
            except subprocess.CalledProcessError as e2:
                print(f"  ❌ Alternative method also failed: {e2}")
                raise
        except Exception as e:
            print(f"  ❌ Error cloning repository: {e}")
            raise
    
    def _apply_test_patch(self, repo_dir: Path, test_patch: str):
        """SKIPPED: Test patch application not needed for code extraction"""
        # SWE-bench harness will apply test patches in Docker
        # We only need the base commit code for extraction
        pass
    
    def _install_dependencies(self, repo_dir: Path):
        """SKIPPED: Package installation not needed for code extraction"""
        # SWE-bench harness will install packages in Docker
        # We only need to read Python files for buggy method extraction
        pass
    
    def restore_original_state(self, repo_dir: Path, original_commit: str):
        """Restore repository to original buggy state"""
        try:
            repo = git.Repo(repo_dir)
            repo.git.reset('--hard', original_commit)
            repo.git.clean('-fdx')
        except Exception as e:
            print(f"  ⚠️  Error restoring state: {e}")


class TestRunner:
    """Run tests for validation"""
    
    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
    
    def run_tests(self, test_files: List[str], timeout: int = 120) -> Dict:
        """Run specified tests"""
        
        if not test_files:
            return {"success": False, "error": "No tests specified"}
        
        # Build test command
        test_args = " ".join(test_files[:10])  # Limit tests
        cmd = f"cd {self.repo_dir} && python -m pytest {test_args} -xvs --tb=short --no-header"
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test execution timeout",
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def extract_error_messages(self, test_output: str) -> List[str]:
        """Extract error messages from test output"""
        errors = []
        
        lines = test_output.split('\n')
        in_error = False
        current_error = []
        
        for line in lines:
            if 'FAILED' in line or 'ERROR' in line:
                in_error = True
                current_error = [line]
            elif in_error:
                if line.strip() == '' or line.startswith('='):
                    if current_error:
                        errors.append('\n'.join(current_error))
                        current_error = []
                    in_error = False
                else:
                    current_error.append(line)
        
        if current_error:
            errors.append('\n'.join(current_error))
        
        return errors[:5]  # Return first 5 errors


class PatchApplicator:
    """Apply and validate patches"""
    
    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.backup_dir = repo_dir / ".backup"
        
    def backup_files(self, files: List[str]):
        """Backup files before applying patch"""
        self.backup_dir.mkdir(exist_ok=True)
        
        for file in files:
            src = self.repo_dir / file
            if src.exists():
                dst = self.backup_dir / file
                dst.parent.mkdir(parents=True, exist_ok=True)
                
                with open(src, 'r') as f:
                    content = f.read()
                with open(dst, 'w') as f:
                    f.write(content)
    
    def apply_patch(self, fixed_files: Dict[str, str]) -> bool:
        """Apply patch by writing fixed files"""
        try:
            # Backup original files
            self.backup_files(list(fixed_files.keys()))
            
            # Write fixed files
            for file_path, content in fixed_files.items():
                full_path = self.repo_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(full_path, 'w') as f:
                    f.write(content)
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error applying patch: {e}")
            self.restore_backup()
            return False
    
    def restore_backup(self):
        """Restore files from backup"""
        if not self.backup_dir.exists():
            return
        
        try:
            for backup_file in self.backup_dir.rglob('*'):
                if backup_file.is_file():
                    rel_path = backup_file.relative_to(self.backup_dir)
                    dst = self.repo_dir / rel_path
                    
                    with open(backup_file, 'r') as f:
                        content = f.read()
                    with open(dst, 'w') as f:
                        f.write(content)
        except Exception as e:
            print(f"  ⚠️  Error restoring backup: {e}")
    
    def cleanup_backup(self):
        """Remove backup directory"""
        import shutil
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)


# ============================================================================
# Utility Functions
# ============================================================================

def parse_test_identifiers(test_list: List[str]) -> List[str]:
    """Convert test identifiers to pytest format"""
    parsed = []
    
    for test in test_list:
        # Check if already in pytest format (contains :: or .py)
        if '::' in test or '.py' in test or '[' in test:
            # Already in pytest format, use as-is
            parsed.append(test)
            continue
        
        # Check if it's a file path
        if '/' in test and not '.' in test.split('/')[-1]:
            # Looks like a directory or file without extension
            parsed.append(test)
            continue
        
        # Format: module.ClassName.test_method or path.to.module.Class.method
        # Convert to: path/to/module.py::ClassName::test_method
        if '.' in test:
            parts = test.split('.')
            
            if len(parts) >= 3:
                # Assume format: path.to.module.ClassName.test_method
                # Last part is method, second-to-last is class, rest is module path
                method = parts[-1]
                classname = parts[-2]
                module_parts = parts[:-2]
                
                # Convert module path to file path
                module_path = '/'.join(module_parts) + '.py'
                
                # Build pytest format
                parsed.append(f"{module_path}::{classname}::{method}")
            elif len(parts) == 2:
                # Format: ClassName.test_method or module.test_function
                # Ambiguous - try both interpretations
                # First try as module.function
                parsed.append(f"{parts[0]}.py::{parts[1]}")
            else:
                # Single part, use as-is
                parsed.append(test)
        else:
            # No dots, use as-is (might be a file or directory)
            parsed.append(test)
    
    return parsed


def extract_changed_files(patch: str) -> List[str]:
    """Extract list of changed files from unified diff patch"""
    import re
    
    files = []
    for match in re.finditer(r'---\s+a/(.*?)\s+\+\+\+\s+b/', patch):
        files.append(match.group(1))
    
    return files


def create_minimal_test_suite(fail_to_pass: List[str], pass_to_pass: List[str]) -> Dict:
    """Create minimal test suite for validation"""
    return {
        "failing_tests": parse_test_identifiers(fail_to_pass[:10]),
        "passing_tests": parse_test_identifiers(pass_to_pass[:20])
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Load dataset
    loader = SWEBenchLoader()
    dataset = loader.load_dataset()
    
    print(f"Loaded {len(dataset)} bug instances")
    
    # Setup first bug
    if dataset:
        repo_mgr = RepositoryManager()
        bug = dataset[0]
        
        print(f"\nSetting up: {bug['instance_id']}")
        repo_dir = repo_mgr.setup_repository(bug)
        
        # Run initial tests
        test_runner = TestRunner(repo_dir)
        test_suite = create_minimal_test_suite(
            bug['FAIL_TO_PASS'],
            bug['PASS_TO_PASS']
        )
        
        print("\nRunning failing tests...")
        result = test_runner.run_tests(test_suite['failing_tests'])
        
        if not result['success']:
            print("✅ Tests fail as expected (buggy version)")
            errors = test_runner.extract_error_messages(result['stdout'])
            print(f"Found {len(errors)} error messages")
        else:
            print("⚠️  Tests pass unexpectedly")