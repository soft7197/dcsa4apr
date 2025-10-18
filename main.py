# main.py
import argparse
import os
import sys
import yaml
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging
import traceback

from src.controller import MainController, BugFixingConfig
from src.evaluation.benchmark_runner import ParallelBenchmarkRunner
from src.preprocessing.vector_db_builder import VectorDBBuilder
from src.evaluation.comparison import ToolComparison

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bugfix.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all prerequisites are met."""
    issues = []
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        issues.append("OpenAI API key not set. Run: export OPENAI_API_KEY='your-key-here'")
    
    # Check Defects4J installation
    try:
        result = subprocess.run(['defects4j', 'info', '-p', 'Chart'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            issues.append("Defects4J not properly installed")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        issues.append("Defects4J not found in PATH")
    
    # Check Java
    try:
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            issues.append("Java not properly installed")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        issues.append("Java not found in PATH")
    
    # Check Python packages
    required_packages = ['openai', 'faiss', 'transformers', 'torch', 'javalang']
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            issues.append(f"Python package '{package}' not installed")
    
    if issues:
        logger.error("Prerequisites check failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("All prerequisites met")
    return True

def load_config(config_file: str) -> BugFixingConfig:
    """Load configuration from YAML file with validation."""
    if not os.path.exists(config_file):
        logger.warning(f"Config file {config_file} not found, using defaults")
        config = BugFixingConfig()
    else:
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Add perfect FL file path if not in config
            if 'perfect_fl_file' not in config_dict:
                config_dict['perfect_fl_file'] = 'data/perfect_fl.json'
            
            config = BugFixingConfig(**config_dict)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            config = BugFixingConfig()
    
    # Validate config
    if config.fl_tool == "perfect" and not os.path.exists(config.perfect_fl_file):
        logger.warning(f"Perfect FL file not found: {config.perfect_fl_file}")
    
    return config

def checkout_defects4j_bug(bug_id: str, working_dir: str = None) -> str:
    """Checkout a Defects4J bug with better error handling."""
    project, bug_num = bug_id.split('-')
    
    if working_dir is None:
        working_dir = f"/tmp/d4j_{bug_id}_{int(time.time())}"
    
    # Clean up existing directory
    if os.path.exists(working_dir):
        import shutil
        try:
            shutil.rmtree(working_dir)
        except Exception as e:
            logger.warning(f"Failed to remove existing directory: {e}")
    
    logger.info(f"Checking out {bug_id} to {working_dir}")
    
    try:
        # Checkout buggy version
        result = subprocess.run([
            "defects4j", "checkout",
            "-p", project,
            "-v", f"{bug_num}b",
            "-w", working_dir
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.error(f"Failed to checkout {bug_id}:")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return None
        
        # Verify checkout
        if not os.path.exists(working_dir):
            logger.error(f"Checkout succeeded but directory not found: {working_dir}")
            return None
        
        logger.info(f"Successfully checked out {bug_id}")
        return working_dir
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while checking out {bug_id}")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking out {bug_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error checking out {bug_id}: {e}")
        return None

def get_failing_tests(bug_id: str, project_path: str) -> List[str]:
    perfect_fl_file = "data/perfect_fl.json"
    if os.path.exists(perfect_fl_file):
        try:
            with open(perfect_fl_file, "r") as file:
                bugs = json.load(file)
            if bug_id in bugs and "trigger_test" in bugs[bug_id]:
                logger.info(f"Found {len(bugs[bug_id]['trigger_test'])} failing tests for {bug_id} from perfect FL")
                return bugs[bug_id]['trigger_test']
            else:
                logger.warning(f"No failing tests found for {bug_id} in perfect FL file")
        except Exception as e:
            logger.error(f"Error reading perfect FL file: {e}")
    # """Get failing tests with better parsing."""
    # try:
    #     # First compile the project
    #     logger.info(f"Compiling {bug_id}...")
    #     compile_result = subprocess.run(
    #         ["defects4j", "compile"],
    #         cwd=project_path,
    #         capture_output=True,
    #         text=True,
    #         timeout=180
    #     )
        
    #     if compile_result.returncode != 0:
    #         logger.warning(f"Compilation issues for {bug_id}: {compile_result.stderr}")
        
    #     # Export failing tests
    #     result = subprocess.run(
    #         ["defects4j", "export", "-p", "tests.trigger"],
    #         cwd=project_path,
    #         capture_output=True,
    #         text=True,
    #         timeout=30
    #     )
        
    #     if result.returncode == 0 and result.stdout:
    #         failing_tests = [t.strip() for t in result.stdout.strip().split('\n') if t.strip()]
    #         logger.info(f"Found {len(failing_tests)} failing tests for {bug_id}")
    #         return failing_tests
        
    #     # Alternative: Try to get from failing_tests file
    #     failing_tests_file = os.path.join(project_path, "failing_tests")
    #     if os.path.exists(failing_tests_file):
    #         with open(failing_tests_file, 'r') as f:
    #             failing_tests = [t.strip() for t in f.readlines() if t.strip()]
    #         return failing_tests
        
    #     logger.warning(f"No failing tests found for {bug_id}")
    #     return []
        
    # except subprocess.TimeoutExpired:
    #     logger.error(f"Timeout getting failing tests for {bug_id}")
    #     return []
    # except Exception as e:
    #     logger.error(f"Error getting failing tests: {e}")
    #     return []

def preprocess_benchmark(benchmark: str, output_dir: str, bug_ids: List[str] = None):
    """Preprocess benchmark with progress tracking."""
    logger.info(f"Preprocessing {benchmark} benchmark...")
    
    builder = VectorDBBuilder()
    
    if benchmark == 'defects4j':
        if bug_ids:
            projects = list(set([bid.split('-')[0] for bid in bug_ids]))
        else:
            projects = ['Chart', 'Closure', 'Lang', 'Math', 'Mockito', 'Time']
        
        total_projects = len(projects)
        for i, project in enumerate(projects, 1):
            logger.info(f"Processing project {project} ({i}/{total_projects})")
            
            # Use a representative bug for project structure
            sample_bug = f"{project}-1"
            project_path = checkout_defects4j_bug(sample_bug)
            
            if project_path and os.path.exists(project_path):
                try:
                    output_path = f"{output_dir}/{project}"
                    os.makedirs(output_path, exist_ok=True)
                    
                    builder.process_project(project_path, output_path)
                    logger.info(f"Successfully processed {project} -> {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing {project}: {e}")
                finally:
                    # Cleanup
                    import shutil
                    try:
                        shutil.rmtree(project_path)
                    except:
                        pass
            else:
                logger.error(f"Failed to checkout sample for {project}")
    
    elif benchmark == 'bugsinpy':
        logger.info("BugsInPy preprocessing not yet fully implemented")
        # Add BugsInPy implementation here

def repair_single_bug(bug_id: str, config: BugFixingConfig, args) -> Dict:
    """Repair a single bug with comprehensive error handling."""
    logger.info(f"="*60)
    logger.info(f"Starting repair for {bug_id}")
    logger.info(f"="*60)
    
    # Initialize result
    result = {
        'bug_id': bug_id,
        'success': False,
        'error': None,
        'patch': None,
        'project_path': None
    }
    
    # Get project name
    try:
        project = bug_id.split('-')[0]
    except:
        logger.error(f"Invalid bug ID format: {bug_id}")
        result['error'] = 'Invalid bug ID format'
        return result
    
    # Checkout bug
    project_path = checkout_defects4j_bug(bug_id)
    if not project_path:
        logger.error(f"Failed to checkout {bug_id}")
        result['error'] = 'Checkout failed'
        return result
    
    result['project_path'] = project_path

    try:
        # Check if vector DB exists
        vector_db_path = f"{args.vector_db_dir}"
        if not os.path.exists(vector_db_path):
            logger.warning(f"Vector DB not found for {project}, creating it now...")
            preprocess_benchmark('defects4j', args.vector_db_dir, [bug_id])
            
            # Re-check
            if not os.path.exists(vector_db_path):
                logger.error(f"Failed to create vector DB for {project}")
                result['error'] = 'Vector DB creation failed'
                return result
        
        # Initialize controller
        controller = MainController(config)
        controller.initialize_components(project_path, vector_db_path)
        
        # Prepare bug info
        bug_info = {
            'bug_id': bug_id,
            'language': 'java' if args.benchmark == 'defects4j' else 'python',
            'project_path': project_path
        }
         
        # Get failing tests
        failing_tests = get_failing_tests(bug_id, project_path)
        if failing_tests:
            bug_info['failing_tests'] = failing_tests
        else:
            logger.warning(f"No failing tests found for {bug_id}, will rely on perfect FL")
        
        # Run repair
        logger.info(f"Running repair with {config.fl_tool} fault localization")
        repair_result = controller.fix_bug(bug_info)
        
        # Update result
        result.update(repair_result)
        
        if result['success']:
            logger.info(f"Successfully generated patch for {bug_id}")
        else:
            logger.warning(f"Failed to generate patch for {bug_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error repairing {bug_id}: {e}")
        logger.error(traceback.format_exc())
        result['error'] = str(e)
        return result

def validate_patch(result: Dict, cleanup: bool = True) -> bool:
    """Validate patch with comprehensive testing."""
    if not result.get('success') or not result.get('patch'):
        logger.info("No patch to validate")
        return False
    
    project_path = result.get('project_path')
    if not project_path or not os.path.exists(project_path):
        logger.error("Project path not found for validation")
        return False
    
    try:
        # First, re-compile after patch
        logger.info("Compiling patched version...")
        compile_result = subprocess.run(
            ['defects4j', 'compile'],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=180
        )
        
        if compile_result.returncode != 0:
            logger.error("Compilation failed after patch")
            return False
        
        # Run all tests
        logger.info("Running all tests for validation...")
        test_result = subprocess.run(
            ['defects4j', 'test'],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        # Check results
        if test_result.returncode == 0:
            logger.info("All tests passed! Patch is valid.")
            return True
        else:
            # Check if only originally failing tests now pass
            failing_tests_file = os.path.join(project_path, "failing_tests")
            if os.path.exists(failing_tests_file):
                with open(failing_tests_file, 'r') as f:
                    still_failing = f.read().strip()
                
                if not still_failing:
                    logger.info("Originally failing tests now pass!")
                    return True
                else:
                    logger.warning(f"Some tests still failing: {still_failing}")
            
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Test validation timed out")
        return False
    except Exception as e:
        logger.error(f"Error validating patch: {e}")
        return False
    finally:
        if cleanup and project_path and os.path.exists(project_path):
            import shutil
            try:
                shutil.rmtree(project_path)
                logger.info(f"Cleaned up {project_path}")
            except:
                logger.warning(f"Failed to cleanup {project_path}")


def process_bug(bug_id, config, args, logger):
    import os, json, traceback, shutil
    result = None
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {bug_id}")
        logger.info(f"{'='*60}")

        result = repair_single_bug(bug_id, config, args)

        # Validate if requested
        if args.validate and result.get('success'):
            is_valid = validate_patch(result, cleanup=not args.no_cleanup)
            result['validated'] = is_valid

        # Save individual result
        output_file = f"{args.output_dir}/{bug_id}_patch.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Result for {bug_id}: {'SUCCESS' if result['success'] else 'FAILED'}")
        if result.get('validated') is not None:
            logger.info(f"Validation: {'PASSED' if result['validated'] else 'FAILED'}")
        logger.info(f"Saved to {output_file}")

        with open("/home/selab/Desktop/automated-bug-fixer/over_log.txt", "a") as f:
            f.write(f"{bug_id}\n")

    except Exception as e:
        logger.error(f"Failed to process {bug_id}: {e}")
        logger.debug(traceback.format_exc())
        result = {
            'bug_id': bug_id,
            'success': False,
            'error': str(e)
        }
    finally:
        # Cleanup if needed
        if result and not args.no_cleanup and 'project_path' in result:
            if os.path.exists(result['project_path']):
                try:
                    shutil.rmtree(result['project_path'])
                except:
                    pass
    return result

def main():
    parser = argparse.ArgumentParser(description='Automated Bug Fixing System')
    
    # Mode selection
    parser.add_argument('--mode', 
                       choices=['preprocess', 'repair', 'evaluate', 'compare', 'validate', 'check'],
                       default='repair', 
                       help='Operation mode')
    
    # Benchmark and bug selection
    parser.add_argument('--benchmark', 
                       choices=['defects4j', 'bugsinpy'], 
                       default='defects4j',
                       help='Benchmark to use')
    parser.add_argument('--bug-id', 
                       type=str,
                       default='Chart-22', 
                       help='Specific bug ID to fix (e.g., Chart-1)')
    parser.add_argument('--bug-list', 
                       type=str, 
                       help='File containing list of bug IDs')
    
    # Configuration
    parser.add_argument('--config', 
                       type=str, 
                       default='configs/default.yaml',
                       help='Configuration file')
    parser.add_argument('--fl-mode',
                       choices=['perfect', 'gzoltar', 'ochiai'],
                       default='perfect',
                       help='Fault localization mode')
    parser.add_argument('--perfect-fl-file',
                       type=str,
                       default='data/perfect_fl.json',
                       help='Path to perfect FL JSON')
    
    # Paths
    parser.add_argument('--output-dir', 
                       type=str, 
                       default='results',
                       help='Output directory')
    parser.add_argument('--vector-db-dir', 
                       type=str, 
                       default='data/vector_db',
                       help='Vector database directory')
    
    # Execution options
    parser.add_argument('--workers', 
                       type=int, 
                       default=4,
                       help='Parallel workers for evaluation')
    parser.add_argument('--max-iterations',
                       type=int,
                       default=10,
                       help='Maximum repair iterations')
    parser.add_argument('--validate',
                       action='store_true',
                       help='Validate patches')
    parser.add_argument('--no-cleanup',
                       action='store_true',
                       help='Keep temporary directories')
    parser.add_argument('--continue-on-error',
                       action='store_true',
                       help='Continue processing even if some bugs fail')
    
    args = parser.parse_args()
    
    # Check mode
    if args.mode == 'check':
        # Just check prerequisites
        if check_prerequisites():
            logger.info("System ready for bug fixing!")
            return 0
        else:
            logger.error("System not ready. Please fix the issues above.")
            return 1
    
    # Check prerequisites for other modes
    if not check_prerequisites():
        logger.error("Prerequisites not met. Run with --mode check for details.")
        return 1
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.fl_mode:
        config.fl_tool = args.fl_mode
    if args.perfect_fl_file:
        config.perfect_fl_file = args.perfect_fl_file
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.vector_db_dir, exist_ok=True)
    
    if args.mode == 'preprocess':
        # Preprocess benchmark to build vector DBs
        bug_ids = None
        if args.bug_list and os.path.exists(args.bug_list):
            with open(args.bug_list, 'r') as f:
                bug_ids = [line.strip() for line in f if line.strip()]
        elif args.bug_id:
            bug_ids = [args.bug_id]
        
        preprocess_benchmark(args.benchmark, args.vector_db_dir, bug_ids)
        
    elif args.mode == 'repair':
    # Repair bug(s)
       
        with open("path/to/perfect_fl.json", "r") as file:
            all_bugs = json.load(file)
        all_results = []
        all_results = [None] * len(bugs_to_repair)  # keep order same
        jobs = getattr(args, "jobs", 16)

        if jobs == 1:
            # Sequential fallback
            for i, bug_id in enumerate(bugs_to_repair):
                all_results[i] = process_bug(bug_id, config, args, logger)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
                future_to_index = {
                    executor.submit(process_bug, bug_id, config, args, logger): i
                    for i, bug_id in enumerate(bugs_to_repair)
                }
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        all_results[idx] = future.result()
                    except Exception as e:
                        bug_id = bugs_to_repair[idx]
                        logger.error(f"Unexpected error in {bug_id}: {e}")
                        all_results[idx] = {
                            'bug_id': bug_id,
                            'success': False,
                            'error': str(e)
                        }

        # =====================
        # Summary (same as before)
        # =====================
        successful = sum(1 for r in all_results if r and r.get('success'))
        validated = sum(1 for r in all_results if r and r.get('validated'))

        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total bugs processed: {len(all_results)}")
        logger.info(f"Successful patches: {successful}/{len(all_results)} ({100*successful/len(all_results):.1f}%)")
        if args.validate and successful > 0:
            logger.info(f"Validated patches: {validated}/{successful} ({100*validated/successful:.1f}%)")

        summary_file = f"{args.output_dir}/repair_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'total': len(all_results),
                'successful': successful,
                'validated': validated if args.validate else None,
                'config': config.__dict__,
                'results': all_results
            }, f, indent=2, default=str)
        logger.info(f"Summary saved to {summary_file}")
        
    elif args.mode == 'evaluate':
        # Run full benchmark evaluation
        runner = ParallelBenchmarkRunner(
            args.benchmark,
            config,
            n_workers=args.workers
        )
        
        # Get bug list
        bug_ids = None
        if args.bug_list and os.path.exists(args.bug_list):
            with open(args.bug_list, 'r') as f:
                bug_ids = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Starting parallel evaluation with {args.workers} workers")
        
        if args.benchmark == 'defects4j':
            results = runner.run_defects4j(bug_ids)
        else:
            results = runner.run_bugsinpy(bug_ids)
        
        # Generate report
        report = runner.generate_report(results)
        
        # Save report
        report_file = f"{args.output_dir}/{args.benchmark}_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\nEvaluation Complete:")
        logger.info(f"Total bugs: {report['total_bugs']}")
        logger.info(f"Plausible patches: {report['plausible_patches']}")
        logger.info(f"Correct patches: {report['correct_patches']}")
        logger.info(f"Success rate: {report['success_rate']:.2%}")
        logger.info(f"Report saved to {report_file}")
        
    elif args.mode == 'validate':
        # Validate existing patches
        if not args.bug_id:
            logger.error("Bug ID required for validation")
            return 1
        
        patch_file = f"{args.output_dir}/{args.bug_id}_patch.json"
        if not os.path.exists(patch_file):
            logger.error(f"Patch file not found: {patch_file}")
            return 1
        
        with open(patch_file, 'r') as f:
            result = json.load(f)
        
        if not result.get('success'):
            logger.error("No successful patch to validate")
            return 1
        
        # Re-checkout the bug for validation
        project_path = checkout_defects4j_bug(args.bug_id)
        if not project_path:
            logger.error("Failed to checkout bug for validation")
            return 1
        
        result['project_path'] = project_path
        
        # Validate
        is_valid = validate_patch(result, cleanup=not args.no_cleanup)
        logger.info(f"Validation result: {'PASSED' if is_valid else 'FAILED'}")
        
        # Update patch file with validation result
        result['validated'] = is_valid
        with open(patch_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
    elif args.mode == 'compare':
        # Compare with other tools
        comparison = ToolComparison()
        
        # Load our results
        our_results_files = list(Path(args.output_dir).glob(f"{args.benchmark}_report_*.json"))
        if our_results_files:
            latest_results = max(our_results_files, key=os.path.getctime)
            comparison.load_results('Our Approach', str(latest_results))
            logger.info(f"Loaded our results from {latest_results}")
        else:
            logger.warning("No results found for our approach")
        
        # Load baseline results
        baseline_dir = 'baselines'
        if os.path.exists(baseline_dir):
            for tool_file in os.listdir(baseline_dir):
                if tool_file.endswith('_results.json'):
                    tool_name = tool_file.replace('_results.json', '')
                    comparison.load_results(tool_name, os.path.join(baseline_dir, tool_file))
                    logger.info(f"Loaded {tool_name} results")
        
        # Generate comparison
        df = comparison.compare_success_rates()
        if not df.empty:
            logger.info("\nTool Comparison:")
            print(df.to_string())
            
            # Save comparison
            comparison_file = f"{args.output_dir}/tool_comparison.csv"
            df.to_csv(comparison_file, index=False)
            logger.info(f"Comparison saved to {comparison_file}")
            
            # Generate plots
            try:
                comparison.plot_comparison()
                logger.info("Comparison plots generated")
            except Exception as e:
                logger.warning(f"Could not generate plots: {e}")
        else:
            logger.warning("Not enough data for comparison")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
