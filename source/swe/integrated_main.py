"""
Complete Integrated Agent-Based APR System
Parallel bug processing with per-patch evaluation and early stopping
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from threading import Lock
import subprocess
import shutil
import time
from pathlib import Path
from typing import Dict
from threading import Lock

# Global lock for coordinating Docker cleanup across workers
_docker_cleanup_lock = Lock()
_last_global_cleanup = {'time': 0}



def cleanup_after_bug_worker(bug_id: str, config):
    """
    Aggressive cleanup after each bug - maximizes space recovery
    Safe for 32 parallel executions
    """
    try:
        # 1. Remove workspace for this bug (SAFE - unique per bug)
        workspace_bug = Path(config.workspace_dir) / bug_id
        if workspace_bug.exists():
            shutil.rmtree(workspace_bug, ignore_errors=True)
            print(f"    🧹 Removed workspace for {bug_id}")
        
        # 2. Remove ALL Docker containers for this bug (SAFE - bug is complete)
        # Pattern: sweb.eval.{bug_id}.* or any container with bug_id
        try:
            # Remove by name pattern
            subprocess.run(
                f"docker ps -a -q --filter 'name=sweb.eval.{bug_id}' | xargs -r docker rm -f",
                shell=True,
                capture_output=True,
                timeout=30
            )
            
            # Also remove by label if SWE-bench uses labels
            subprocess.run(
                f"docker ps -a -q --filter 'label=instance_id={bug_id}' | xargs -r docker rm -f",
                shell=True,
                capture_output=True,
                timeout=30
            )
            
            print(f"    🧹 Removed Docker containers for {bug_id}")
        except subprocess.TimeoutExpired:
            print(f"    ⚠️ Container removal timeout (non-critical)")
        except Exception as e:
            print(f"    ⚠️ Container removal warning: {e}")
        
        # 3. Remove evaluation logs for this bug (SAVES MOST SPACE)
        # These logs can be 100s of MB per bug
        # 4. Remove dangling volumes (if any)
        try:
            subprocess.run(
                "docker volume prune -f",
                shell=True,
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            pass  # Non-critical
        
    except Exception as e:
        # Silent failure - cleanup is non-critical for correctness
        pass


# ============================================================================
# OPTIONAL: Add space monitoring
# ============================================================================

def check_disk_space_and_warn():
    """Check disk space and warn if low"""
    try:
        result = subprocess.run(
            "df -h / | tail -1 | awk '{print $5}' | sed 's/%//'",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        usage_percent = int(result.stdout.strip())
        
        if usage_percent > 98:
            print(f"\n⚠️  WARNING: Disk usage at {usage_percent}%!")
            print(f"    Running emergency cleanup...")
            
            # Emergency cleanup
            subprocess.run("docker system prune -a -f --volumes", shell=True, timeout=120)
            subprocess.run("rm -rf /home/selab/Desktop/swe_exp/logs/run_evaluation/*", shell=True, timeout=60)
            
            print(f"    Emergency cleanup complete")
        
        return usage_percent
        
    except Exception as e:
        return None



def periodic_docker_cleanup():
    """
    Global Docker cleanup - coordinated across all workers
    Only ONE worker executes this at a time
    Runs at most once per 60 seconds
    """
    current_time = time.time()
    
    # Check if cleanup is needed (60 second cooldown)
    if current_time - _last_global_cleanup['time'] < 60:
        return  # Too soon, skip
    
    # Try to acquire lock (non-blocking)
    if not _docker_cleanup_lock.acquire(blocking=False):
        return  # Another worker is already cleaning, skip
    
    try:
        # Update timestamp BEFORE cleanup to prevent multiple workers
        _last_global_cleanup['time'] = current_time
        
        # Prune dangling images
        subprocess.run(
            "docker image prune -f",
            shell=True,
            capture_output=True,
            timeout=90
        )
        
        # Prune stopped containers (belt and suspenders)
        subprocess.run(
            "docker container prune -f",
            shell=True,
            capture_output=True,
            timeout=60
        )
        
    except Exception as e:
        pass  # Non-critical
    finally:
        _docker_cleanup_lock.release()

def save_bug_complete_results(bug_id: str, pipeline, repair_result: Dict, config):
    """
    Save complete results for a bug including all iteration details
    Thread-safe for parallel execution
    """
    try:
        # Create bug-specific results directory
        bug_results_dir = Path(config.results_dir) / "bug_results" / bug_id
        bug_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Compile complete bug data
        complete_results = {
            'bug_id': bug_id,
            'timestamp': time.time(),
            'success': repair_result.get('success', False),
            'total_time': repair_result.get('time', 0),
            'final_iteration': repair_result.get('iteration', 0),
            'total_patches_evaluated': repair_result.get('total_patches_evaluated', 0),
            
            # Bug details
            'problem_statement': pipeline.context_pool.problem_statement[:500],
            'num_buggy_methods': len(pipeline.context_pool.buggy_methods),
            'buggy_methods': list(pipeline.context_pool.buggy_methods.keys()),
            'failing_tests': pipeline.context_pool.failing_tests,
            
            # Final patch (if successful)
            'final_hypothesis': repair_result.get('hypothesis', ''),
            'final_patch': repair_result.get('model_patch', ''),
            
            # All iterations
            'iterations': [],
            
            # Evaluation result
            'eval_result': repair_result.get('eval_result', {})
        }
        
        # Add iteration details WITH TOOL EXECUTIONS
        for iter_log in pipeline.context_pool.iteration_logs:
            iteration_data = {
                'iteration': iter_log.iteration,
                'timestamp': iter_log.timestamp,
                'num_hypotheses': len(iter_log.hypotheses),
                'hypotheses_summary': [
                    {
                        'patch_id': h.get('patch_id', ''),
                        'summary': h.get('summary', ''),
                        'status': h.get('eval_result', {}).get('status', 'unknown'),
                        'resolved': h.get('eval_result', {}).get('resolved', False)
                    }
                    for h in iter_log.hypotheses
                ],
                'success': iter_log.success,
                'successful_hypothesis': iter_log.successful_hypothesis,
                'prompt_length': len(iter_log.prompt),
                'error_messages': iter_log.error_messages[:3],  # Keep first 3 errors
                'tool_executions': getattr(iter_log, 'tool_executions', [])  # Get tool executions
            }
            complete_results['iterations'].append(iteration_data)
        
        # Save complete results
        results_file = bug_results_dir / "complete_results.json"
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"    💾 [Worker] Saved complete results: {results_file}")
        
        # Also save successful patch separately if found
        if repair_result.get('success') and repair_result.get('model_patch'):
            patch_file = bug_results_dir / "successful_patch.diff"
            with open(patch_file, 'w') as f:
                f.write(repair_result['model_patch'])
            print(f"    💾 [Worker] Saved successful patch: {patch_file}")
        
    except Exception as e:
        print(f"    ⚠️ Warning: Could not save complete results for {bug_id}: {e}")

# Import all components
from main import (
    Config, BugInstance, ContextPool,
    ToolSuite, ContextUpdaterAgent, GeneratorAgent, 
    OverfittingDetectorAgent, RepairPipeline, Hypothesis,
    SWEBenchEvaluator, BuggyMethodExtractor
)
from dataset_loader import (
    SWEBenchLoader, RepositoryManager, TestRunner, 
    PatchApplicator, create_minimal_test_suite, extract_changed_files
)


class EnhancedRepairPipeline(RepairPipeline):
    """Enhanced repair pipeline with immediate per-patch evaluation"""
    
    def __init__(self, bug: BugInstance, repo_path: str):
        super().__init__(bug, repo_path)
        
    def initialize(self):
        """Enhanced initialization"""
        super().initialize()
        print("  ✅ Ready for patch generation and evaluation")

def process_single_bug_worker(bug_data: Dict, config, bug_index: int, total_bugs: int) -> Dict:
    """
    Modified worker function with aggressive cleanup and space monitoring
    """
    
    start_time = time.time()
    bug_id = bug_data['instance_id']
    
    print(f"\n{'='*80}")
    print(f"[Worker {bug_index}/{total_bugs}] Processing: {bug_id}")
    
    # CHECK DISK SPACE BEFORE STARTING
    # disk_usage = check_disk_space_and_warn()
    # if disk_usage:
    #     print(f"[Worker {bug_index}] Disk usage: {disk_usage}%")
    
    print(f"{'='*80}")
    
    try:
        # Convert to BugInstance
        from main import BugInstance
        import json
        
        bug = BugInstance(
            instance_id=bug_data['instance_id'],
            repo=bug_data['repo'],
            base_commit=bug_data['base_commit'],
            problem_statement=bug_data['problem_statement'],
            hints_text=bug_data.get('hints_text', ''),
            test_patch=bug_data['test_patch'],
            patch=bug_data['patch'],
            version=bug_data['version'],
            fail_to_pass=json.loads(bug_data.get('FAIL_TO_PASS', [])) if isinstance(bug_data.get('FAIL_TO_PASS'), str) else bug_data.get('FAIL_TO_PASS', []),
            pass_to_pass=json.loads(bug_data.get('PASS_TO_PASS', [])) if isinstance(bug_data.get('PASS_TO_PASS'), str) else bug_data.get('PASS_TO_PASS', [])
        )
        
        # Setup repository
        print(f"\n📦 [Worker {bug_index}] Setting up repository: {bug.repo}")
        from dataset_loader import RepositoryManager
        
        repo_manager = RepositoryManager(config.workspace_dir)
        repo_dir = repo_manager.setup_repository(bug_data)
        
        # Create repair pipeline
        print(f"\n🤖 [Worker {bug_index}] Initializing repair pipeline...")
        from main import RepairPipeline
        
        pipeline = RepairPipeline(bug=bug, repo_path=str(repo_dir))
        pipeline.initialize()
        
        # Run repair
        print(f"\n🔧 [Worker {bug_index}] Starting repair...")
        repair_result = pipeline.run()
        
        # SAVE COMPLETE RESULTS
        save_bug_complete_results(bug_id, pipeline, repair_result, config)
        
        # ===== AGGRESSIVE CLEANUP AFTER COMPLETION =====
        # print(f"\n🧹 [Worker {bug_index}] Cleaning up after {bug_id}...")
        # cleanup_after_bug_worker(bug_id, config)
        
        # # PERIODIC GLOBAL CLEANUP
        # periodic_docker_cleanup()
        
        # CHECK DISK SPACE AFTER CLEANUP
        # disk_usage_after = check_disk_space_and_warn()
        # if disk_usage_after:
        #     print(f"[Worker {bug_index}] Disk usage after cleanup: {disk_usage_after}%")
        
        # Compile result summary
        result = {
            'bug_id': bug_id,
            'repo': bug.repo,
            'success': repair_result.get('success', False),
            'iteration': repair_result.get('iteration', 0),
            'hypothesis': repair_result.get('hypothesis', ''),
            'time_elapsed': time.time() - start_time,
            'num_buggy_methods': len(pipeline.context_pool.buggy_methods),
            'total_patches_evaluated': repair_result.get('total_patches_evaluated', 0),
            'total_iterations': len(pipeline.context_pool.iteration_logs),
            'worker_id': bug_index,
            'results_saved': True,
            'disk_usage_before': None, # disk_usage ,
            'disk_usage_after': None # disk_usage_after
        }
        
        status = "✅ RESOLVED" if result['success'] else "❌ FAILED"
        print(f"\n[Worker {bug_index}] {status} - {bug_id}")
        print(f"[Worker {bug_index}] Time: {result['time_elapsed']:.1f}s")
        
        return result
        
    except Exception as e:
        print(f"  ❌ [Worker {bug_index}] Error: {e}")
        
        # # AGGRESSIVE CLEANUP even on error
        # cleanup_after_bug_worker(bug_id, config)
        # periodic_docker_cleanup()
        
        return {
            'bug_id': bug_id,
            'repo': bug_data.get('repo', 'unknown'),
            'success': False,
            'error': str(e)[:200],
            'time_elapsed': time.time() - start_time,
            'worker_id': bug_index,
            'results_saved': False
        }


def process_single_bug_worker_old(bug_data: Dict, config: Config, bug_index: int, total_bugs: int) -> Dict:
    """
    Worker function to process a single bug in parallel
    This runs in a separate process
    """
    
    start_time = time.time()
    bug_id = bug_data['instance_id']
    
    print(f"\n{'='*80}")
    print(f"[Worker {bug_index}/{total_bugs}] Processing: {bug_id}")
    print(f"{'='*80}")
    
    # Convert to BugInstance
    bug = BugInstance(
        instance_id=bug_data['instance_id'],
        repo=bug_data['repo'],
        base_commit=bug_data['base_commit'],
        problem_statement=bug_data['problem_statement'],
        hints_text=bug_data.get('hints_text', ''),
        test_patch=bug_data['test_patch'],
        patch=bug_data['patch'],
        version=bug_data['version'],
        fail_to_pass=json.loads(bug_data.get('FAIL_TO_PASS', [])),
        pass_to_pass=json.loads(bug_data.get('PASS_TO_PASS', []))
    )
    
    # Setup repository
    print(f"\n📦 [Worker {bug_index}] Setting up repository: {bug.repo}")
    try:
        repo_manager = RepositoryManager(config.workspace_dir)
        repo_dir = repo_manager.setup_repository(bug_data)
    except Exception as e:
        print(f"  ❌ [Worker {bug_index}] Repository setup failed: {e}")
        return {
            'bug_id': bug_id,
            'repo': bug.repo,
            'success': False,
            'error': f'Repository setup failed: {str(e)[:200]}',
            'time_elapsed': time.time() - start_time,
            'worker_id': bug_index
        }
    
    # Create repair pipeline
    print(f"\n🤖 [Worker {bug_index}] Initializing repair pipeline...")
    try:
        pipeline = EnhancedRepairPipeline(
            bug=bug,
            repo_path=str(repo_dir)
        )
        
        # Initialize
        pipeline.initialize()
        
        # Run repair (generates, evaluates immediately, stops on plausible)
        print(f"\n🔧 [Worker {bug_index}] Starting repair with per-patch evaluation...")
        repair_result = pipeline.run()
        
    except Exception as e:
        print(f"  ❌ [Worker {bug_index}] Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'bug_id': bug_id,
            'repo': bug.repo,
            'success': False,
            'error': f'Pipeline failed: {str(e)[:200]}',
            'time_elapsed': time.time() - start_time,
            'worker_id': bug_index
        }
    
    # Compile result
    result = {
        'bug_id': bug_id,
        'repo': bug.repo,
        'success': repair_result.get('success', False),
        'iteration': repair_result.get('iteration', 0),
        'hypothesis': repair_result.get('hypothesis', ''),
        'time_elapsed': time.time() - start_time,
        'num_buggy_methods': len(pipeline.context_pool.buggy_methods),
        'total_patches_evaluated': repair_result.get('total_patches_evaluated', 0),
        'total_iterations': len(pipeline.context_pool.iteration_logs),
        'worker_id': bug_index
    }
    
    status = "✅ RESOLVED" if result['success'] else "❌ FAILED"
    print(f"\n[Worker {bug_index}] {status} - {bug_id}")
    print(f"[Worker {bug_index}] Time: {result['time_elapsed']:.1f}s")
    
    return result


class ParallelIntegratedRepairSystem:
    """Complete integrated repair system with parallel bug processing"""
    
    def __init__(self, config: Config, max_workers: int = None):
        self.config = config
        self.loader = SWEBenchLoader()
        self.results = []
        self.results_lock = Lock()
        
        # Determine number of workers
        if max_workers is None:
            # Use 50% of CPU cores by default (each bug is resource-intensive)
            cpu_count = mp.cpu_count()
            self.max_workers = max(1, cpu_count // 2)
        else:
            self.max_workers = max_workers
        
        print(f"🔧 Parallel processing with {self.max_workers} workers")
        
    def run_full_evaluation(self, num_bugs: Optional[int] = None):
        """Run complete evaluation with parallel bug processing"""
        
        print("="*80)
        print("AGENT-BASED AUTOMATED PROGRAM REPAIR SYSTEM")
        print("Parallel Bug Processing with Per-Patch Evaluation")
        print("="*80)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration:")
        print(f"  - Model: {self.config.model_generator}")
        print(f"  - Max Iterations: {self.config.max_iterations}")
        print(f"  - Patches per Iteration: 10")
        print(f"  - Parallel Workers: {self.max_workers}")
        print(f"  - Evaluation: Per-patch with SWE-bench")
        print(f"  - Early stopping: Yes (on first plausible patch)")
        print()
        
        # Load dataset
        dataset = self.loader.load_dataset()
        
        if not dataset:
            print("❌ No dataset loaded. Exiting.")
            return
        
        print(f"✅ Loaded {len(dataset)} bug instances")
        
        # Limit number of bugs if specified
        if num_bugs:
            dataset = dataset[:num_bugs]
            #print(f"Processing first {num_bugs} bugs\n")
        
        
        
        not_evaluated = ['sympy__sympy-14317', 'sympy__sympy-14396', 'sympy__sympy-14774', 'sympy__sympy-14817', 'sympy__sympy-15011', 'sympy__sympy-15308', 'sympy__sympy-15345', 'sympy__sympy-15346', 'sympy__sympy-15609', 'sympy__sympy-15678', 'sympy__sympy-16106', 'sympy__sympy-16281', 'sympy__sympy-16503', 'sympy__sympy-16792', 'sympy__sympy-16988', 'sympy__sympy-17022', 'sympy__sympy-17139', 'sympy__sympy-17630', 'sympy__sympy-17655', 'sympy__sympy-18057', 'sympy__sympy-18087', 'sympy__sympy-18189', 'sympy__sympy-18199', 'sympy__sympy-18532', 'sympy__sympy-18621', 'sympy__sympy-18698', 'sympy__sympy-18835', 'sympy__sympy-19007', 'sympy__sympy-19254', 'sympy__sympy-19487', 'sympy__sympy-20049', 'sympy__sympy-20154', 'sympy__sympy-20212', 'sympy__sympy-20322', 'sympy__sympy-20442', 'sympy__sympy-20590', 'sympy__sympy-20639', 'sympy__sympy-21055', 'sympy__sympy-21171', 'sympy__sympy-21379', 'sympy__sympy-21612', 'sympy__sympy-21614', 'sympy__sympy-21627', 'sympy__sympy-21847', 'sympy__sympy-22005', 'sympy__sympy-22714', 'sympy__sympy-22840', 'sympy__sympy-23117', 'sympy__sympy-23191', 'sympy__sympy-23262', 'sympy__sympy-24066', 'sympy__sympy-24102', 'sympy__sympy-24152', 'sympy__sympy-24213', 'sympy__sympy-24909']       # dataset = dataset[:50]
        # plausibel_patches = ['astropy__astropy-12907', 'astropy__astropy-14995', 'astropy__astropy-6938', 'django__django-11001', 'django__django-11133', 'django__django-11179', 'django__django-11583', 'django__django-11848', 'django__django-11999', 'django__django-12113', 'django__django-12286', 'django__django-12700', 'django__django-12708', 'django__django-12915', 'django__django-13028', 'django__django-13158']
        
        
        dataset = [d for d in dataset if d['instance_id']  in not_evaluated]
        # dataset = [d for d in dataset if d['instance_id'] == 'pallets__flask-4992']
        
        
        print(f"Processing {len(dataset)} bugs\n")
        # dataset = dataset[:1]
        # Setup results directory
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Process bugs in parallel
        print(f"\n🚀 Starting parallel processing of {len(dataset)} bugs...")
        print(f"   Workers: {self.max_workers}")
        print(f"   Expected speedup: ~{min(self.max_workers, len(dataset))}x")
        print()
        
        start_time = time.time()
        completed = 0
        total_bugs = len(dataset)
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=16) as executor:
            # Submit all bugs
            future_to_bug = {
                executor.submit(
                    process_single_bug_worker, 
                    bug_data, 
                    self.config,
                    idx,
                    total_bugs
                ): (idx, bug_data['instance_id'])
                for idx, bug_data in enumerate(dataset, 1)
            }
            
            # Process completed bugs as they finish
            for future in as_completed(future_to_bug):
                bug_idx, bug_id = future_to_bug[future]
                completed += 1
                
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    # Print progress
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (total_bugs - completed) * avg_time
                    
                    status = "✅" if result.get('success') else "❌"
                    print(f"\n{'='*80}")
                    print(f"Progress: {completed}/{total_bugs} ({completed/total_bugs*100:.1f}%)")
                    print(f"{status} {bug_id}: {result.get('success', False)}")
                    print(f"⏱️  Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")
                    print(f"{'='*80}")
                    
                    # Save intermediate results
                    self._save_results(results_dir / f"results_{timestamp}.json")
                    
                except Exception as e:
                    print(f"\n❌ ERROR processing {bug_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    self.results.append({
                        'bug_id': bug_id,
                        'success': False,
                        'error': str(e),
                        'worker_id': bug_idx
                    })
        
        total_time = time.time() - start_time
        
        # Print final summary
        total_resolved = sum(1 for r in self.results if r.get('success', False))
        self._print_summary(total_resolved, len(dataset), total_time)
        
        # Save final results
        final_results_path = results_dir / f"final_results_{timestamp}.json"
        self._save_results(final_results_path)
        print(f"\n📊 Final results saved to: {final_results_path}")
        print(f"📁 Iteration logs: {results_dir}/iteration_logs/")
        print(f"📁 Per-patch predictions: {self.config.predictions_dir}/")
   

    def _save_results(self, filepath: Path):
        """Save results to JSON file (thread-safe)"""
        
        with self.results_lock:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_bugs': len(self.results),
                'resolved': sum(1 for r in self.results if r.get('success', False)),
                'failed': sum(1 for r in self.results if not r.get('success', False)),
                'results': sorted(self.results, key=lambda x: x.get('worker_id', 0))
            }
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
    
    def _print_summary(self, resolved: int, total: int, total_time: float):
        """Print final summary"""
        
        print("\n" + "="*80)
        print("FINAL EVALUATION SUMMARY")
        print("="*80)
        print(f"\n📊 Overall Results:")
        print(f"   Total Bugs:     {total}")
        print(f"   Resolved:       {resolved}")
        print(f"   Failed:         {total - resolved}")
        print(f"   Success Rate:   {resolved/total*100:.1f}%")
        print(f"\n⏱️  Time Statistics:")
        print(f"   Total Time:     {total_time/60:.1f} minutes")
        print(f"   Avg per Bug:    {total_time/total:.1f} seconds")
        print(f"   Parallelization: {self.max_workers} workers")
        print(f"   Speedup:        ~{min(self.max_workers, total)}x (theoretical)")
        
        # Iteration breakdown
        iteration_counts = {}
        for result in self.results:
            if result.get('success'):
                iter_num = result.get('iteration', 0)
                iteration_counts[iter_num] = iteration_counts.get(iter_num, 0) + 1
        
        if iteration_counts:
            print(f"\n🔄 Fixes by Iteration:")
            for iter_num in sorted(iteration_counts.keys()):
                count = iteration_counts[iter_num]
                print(f"   Iteration {iter_num}:  {count} bugs ({count/resolved*100:.1f}%)")
        
        # Time distribution
        if self.results:
            times = [r.get('time_elapsed', 0) for r in self.results if r.get('time_elapsed')]
            if times:
                print(f"\n⏱️  Time Distribution:")
                print(f"   Min:     {min(times)/60:.1f} minutes")
                print(f"   Max:     {max(times)/60:.1f} minutes")
                print(f"   Median:  {sorted(times)[len(times)//2]/60:.1f} minutes")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Parallel Agent-Based APR with Per-Patch Evaluation'
    )
    
    parser.add_argument(
        '--num-bugs', '-n',
        type=int,
        default=None,
        help='Number of bugs to process (default: all)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU_count/2)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use (default: gpt-4o)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum iterations per bug (default: 5)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Timeout per bug in seconds (default: 3600)'
    )
    
    parser.add_argument(
        '--workspace',
        type=str,
        default='./workspace',
        help='Workspace directory (default: ./workspace)'
    )
    
    parser.add_argument(
        '--results',
        type=str,
        default='./results',
        help='Results directory (default: ./results)'
    )
    
    parser.add_argument(
        '--predictions',
        type=str,
        default='./predictions',
        help='Predictions directory (default: ./predictions)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr add it to a .env file")
        sys.exit(1)
    
    # Validate workers
    if args.workers:
        cpu_count = mp.cpu_count()
        if args.workers > cpu_count:
            print(f"⚠️  Warning: Requested {args.workers} workers but only {cpu_count} CPUs available")
            print(f"   Using {cpu_count} workers instead")
            args.workers = cpu_count
    
    # Create config
    config = Config(
        model_generator=args.model,
        model_context_updater=args.model,
        model_overfitting=args.model,
        max_iterations=args.max_iterations,
        timeout_per_bug=args.timeout,
        workspace_dir=args.workspace,
        results_dir=args.results,
        predictions_dir=args.predictions,
        use_swebench_eval=True
    )
    
    # Create parallel system
    system = ParallelIntegratedRepairSystem(config, max_workers=args.workers)
    
    # Run evaluation
    try:
        system.run_full_evaluation(num_bugs=args.num_bugs)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Saving partial results...")
        
        results_dir = Path(config.results_dir)
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        system._save_results(results_dir / f"partial_results_{timestamp}.json")
        
        print("✅ Partial results saved")
        print("\nProgress saved. You can analyze partial results in:")
        print(f"  - {results_dir}/partial_results_{timestamp}.json")
        print(f"  - {results_dir}/iteration_logs/")
        print(f"  - {config.predictions_dir}/")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()