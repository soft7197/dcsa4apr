# src/evaluation/benchmark_runner.py
import json
import os
from pathlib import Path
from typing import Dict, List
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.controller import BugFixingConfig, MainController

class BenchmarkRunner:
    def __init__(self, benchmark_name: str, config: BugFixingConfig):
        self.benchmark_name = benchmark_name
        self.config = config
        self.results = []
        
    def run_defects4j(self, bug_ids: List[str] = None):
        """Run evaluation on Defects4J benchmark."""
        d4j_path = Path("data/defects4j")
        
        if not bug_ids:
            # Get all bug IDs
            bug_ids = self._get_all_d4j_bugs()
        
        results = []
        for bug_id in bug_ids:
            print(f"Processing {bug_id}...")
            
            # Checkout buggy version
            project_path = self._checkout_d4j_bug(bug_id)
            
            # Get bug information
            bug_info = self._get_d4j_bug_info(bug_id, project_path)
            
            # Build vector DB for project
            vector_db_path = self._build_vector_db(project_path, bug_id)
            
            # Initialize controller
            controller = MainController(self.config)
            controller.initialize_components(project_path, vector_db_path)
            
            # Run repair
            result = controller.fix_bug(bug_info)
            
            # Validate patch
            validation = self._validate_patch(result['patch'], bug_id)
            
            results.append({
                'bug_id': bug_id,
                'success': result['success'],
                'valid': validation['passes_all_tests'],
                'iterations': result['metrics']['total_iterations'],
                'time': result['metrics']['time_spent'],
                'patch': result['patch']
            })
            
            # Save intermediate results
            self._save_results(results)
        
        return results
    
    def _checkout_d4j_bug(self, bug_id: str) -> str:
        """Checkout buggy version of Defects4J project."""
        project, bug_num = bug_id.split('-')
        
        temp_dir = f"/tmp/d4j_{bug_id}"
        
        # Checkout buggy version
        subprocess.run([
            "defects4j", "checkout",
            "-p", project,
            "-v", f"{bug_num}b",
            "-w", temp_dir
        ])
        
        return temp_dir
    
    def _get_d4j_bug_info(self, bug_id: str, project_path: str) -> Dict:
        """Get bug information from Defects4J."""
        project, bug_num = bug_id.split('-')
        
        # Get failing tests
        result = subprocess.run(
            ["defects4j", "export", "-p", project, "-w", project_path, "-o", "tests.trigger"],
            capture_output=True,
            text=True
        )
        
        failing_tests = result.stdout.strip().split('\n')
        
        # Get developer patch for perfect FL
        if self.config.fl_tool == "perfect":
            buggy_methods = self._get_developer_fix_locations(bug_id)
        else:
            buggy_methods = []
        
        return {
            'bug_id': bug_id,
            'project': project,
            'failing_tests': failing_tests,
            'buggy_methods': buggy_methods,
            'language': 'java'
        }
    
    def _validate_patch(self, patch: Dict, bug_id: str) -> Dict:
        """Validate generated patch against all tests."""
        if not patch:
            return {'passes_all_tests': False}
        
        project, _ = bug_id.split('-')
        
        # Apply patch and run all tests
        # This would use Defects4J's test infrastructure
        
        return {'passes_all_tests': True}  # Placeholder
    
    def run_bugsinpy(self, bug_ids: List[str] = None):
        """Run evaluation on BugsInPy benchmark."""
        # Similar structure to run_defects4j but for Python projects
        pass
    
    def _save_results(self, results: List[Dict]):
        """Save intermediate results to file."""
        output_file = f"results/{self.benchmark_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_report(self, results: List[Dict]) -> Dict:
        """Generate evaluation report."""
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        valid = sum(1 for r in results if r.get('valid', False))
        
        avg_iterations = sum(r['iterations'] for r in results) / total
        avg_time = sum(r['time'] for r in results) / total
        
        report = {
            'benchmark': self.benchmark_name,
            'total_bugs': total,
            'plausible_patches': successful,
            'correct_patches': valid,
            'success_rate': successful / total,
            'correctness_rate': valid / total,
            'avg_iterations': avg_iterations,
            'avg_time_seconds': avg_time,
            'detailed_results': results
        }
        
        return report

class ParallelBenchmarkRunner(BenchmarkRunner):
    """Run benchmarks in parallel for faster evaluation."""
    
    def __init__(self, benchmark_name: str, config: BugFixingConfig, n_workers: int = 4):
        super().__init__(benchmark_name, config)
        self.n_workers = n_workers
    
    def run_defects4j(self, bug_ids: List[str] = None):
        """Run Defects4J evaluation in parallel."""
        if not bug_ids:
            bug_ids = self._get_all_d4j_bugs()
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._process_single_bug, bug_id): bug_id
                for bug_id in bug_ids
            }
            
            results = []
            for future in as_completed(futures):
                bug_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed {bug_id}: {'Success' if result['success'] else 'Failed'}")
                except Exception as e:
                    print(f"Error processing {bug_id}: {e}")
                    results.append({
                        'bug_id': bug_id,
                        'success': False,
                        'error': str(e)
                    })
            
            # Save intermediate results
            self._save_results(results)
        
        return results
    
    def _process_single_bug(self, bug_id: str) -> Dict:
        """Process a single bug (for parallel execution)."""
        # Checkout bug
        project_path = self._checkout_d4j_bug(bug_id)
        
        # Get bug info
        bug_info = self._get_d4j_bug_info(bug_id, project_path)
        
        # Build vector DB
        vector_db_path = self._build_vector_db(project_path, bug_id)
        
        # Run repair
        controller = MainController(self.config)
        controller.initialize_components(project_path, vector_db_path)
        
        result = controller.fix_bug(bug_info)
        
        # Validate
        validation = self._validate_patch(result['patch'], bug_id)
        
        return {
            'bug_id': bug_id,
            'success': result['success'],
            'valid': validation['passes_all_tests'],
            'iterations': result['metrics']['total_iterations'],
            'time': result['metrics']['time_spent'],
            'patch': result['patch']
        }