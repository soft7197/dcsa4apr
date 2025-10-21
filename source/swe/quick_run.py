#!/usr/bin/env python3
"""
Quick Run Script - Start APR System Immediately
Usage: python quick_run.py
"""

import os
import sys
import subprocess
from pathlib import Path


def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    else:
        print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        issues.append("OPENAI_API_KEY not set")
    else:
        print("  ✅ OpenAI API key configured")
    
    # Check required packages
    required_packages = [
        'openai', 'sentence_transformers', 'faiss', 
        'numpy', 'torch', 'transformers', 'datasets', 'pytest'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
    
    if missing:
        issues.append(f"Missing packages: {', '.join(missing)}")
    
    return issues


def setup_environment():
    """Setup environment if needed"""
    print("\n📦 Setting up environment...")
    
    # Create directories
    dirs = ['workspace', 'results', 'faiss_index', 'datasets']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"  ✅ Created {d}/")
    
    # Check for dataset
    dataset_path = Path('datasets/swebench_lite.json')
    if not dataset_path.exists():
        print("\n📥 Dataset not found. Will be downloaded on first run.")
    else:
        print(f"  ✅ Dataset found ({dataset_path})")


def install_requirements():
    """Install requirements if needed"""
    print("\n📦 Installing requirements...")
    
    req_file = Path('requirements.txt')
    if not req_file.exists():
        print("  ⚠️  requirements.txt not found")
        print("  Creating basic requirements.txt...")
        
        requirements = """openai>=1.12.0
sentence-transformers>=2.3.0
faiss-cpu>=1.7.4
numpy>=1.24.0
torch>=2.0.0
transformers>=4.36.0
datasets>=2.16.0
pytest>=7.4.0
pytest-cov>=4.1.0
GitPython>=3.1.40
astroid>=3.0.0"""
        
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        print("  ✅ Created requirements.txt")
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'],
            check=True
        )
        print("  ✅ Requirements installed")
    except subprocess.CalledProcessError:
        print("  ⚠️  Some packages failed to install")
        print("  You may need to install them manually")


def get_user_choice():
    """Get user's choice for what to run"""
    print("\n" + "="*60)
    print("AGENT-BASED APR SYSTEM - QUICK START")
    print("="*60)
    print("\nWhat would you like to do?")
    print("\n1. Quick Test (5 bugs, ~10 minutes)")
    print("2. Medium Test (20 bugs, ~30 minutes)")
    print("3. Full Evaluation (300 bugs, ~6 hours)")
    print("4. Custom Configuration")
    print("5. Check System Status Only")
    print("6. Exit")
    
    while True:
        choice = input("\nEnter choice (1-6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            return choice
        print("Invalid choice. Please enter 1-6.")


def run_system(num_bugs=None, max_iter=5):
    """Run the APR system"""
    
    cmd = [sys.executable, 'integrated_main.py']
    
    if num_bugs:
        cmd.extend(['--num-bugs', str(num_bugs)])
    
    cmd.extend(['--max-iterations', str(max_iter)])
    
    print(f"\n🚀 Starting APR system...")
    print(f"Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to interrupt (progress will be saved)\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Partial results have been saved")


def main():
    """Main entry point"""
    
    print("="*60)
    print("APR SYSTEM QUICK START")
    print("="*60)
    
    # Check if main files exist
    required_files = ['main.py', 'dataset_loader.py', 'integrated_main.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"\n❌ Error: Missing required files: {', '.join(missing_files)}")
        print("\nPlease ensure all system files are in the current directory:")
        for f in required_files:
            print(f"  - {f}")
        sys.exit(1)
    
    # Check requirements
    issues = check_requirements()
    
    if issues:
        print(f"\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        
        if "OPENAI_API_KEY not set" in issues:
            api_key = input("\nEnter your OpenAI API key (or press Enter to exit): ").strip()
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
                print("✅ API key set for this session")
                # Re-check
                issues = check_requirements()
            else:
                print("\n❌ Cannot proceed without API key")
                sys.exit(1)
        
        if any("Missing packages" in issue for issue in issues):
            response = input("\nInstall missing packages? (y/n): ").strip().lower()
            if response == 'y':
                install_requirements()
            else:
                print("\n❌ Cannot proceed without required packages")
                sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Get user choice
    choice = get_user_choice()
    
    if choice == '1':
        print("\n📋 Quick Test Configuration:")
        print("  - 5 bugs")
        print("  - 5 iterations max")
        print("  - Expected time: ~10 minutes")
        input("\nPress Enter to start...")
        run_system(num_bugs=5, max_iter=5)
        
    elif choice == '2':
        print("\n📋 Medium Test Configuration:")
        print("  - 20 bugs")
        print("  - 5 iterations max")
        print("  - Expected time: ~30 minutes")
        input("\nPress Enter to start...")
        run_system(num_bugs=20, max_iter=5)
        
    elif choice == '3':
        print("\n📋 Full Evaluation Configuration:")
        print("  - 300 bugs (all)")
        print("  - 5 iterations max")
        print("  - Expected time: ~6 hours")
        print("\n⚠️  This will use significant API credits!")
        response = input("\nAre you sure? (yes/no): ").strip().lower()
        if response == 'yes':
            run_system(max_iter=5)
        else:
            print("Cancelled.")
            
    elif choice == '4':
        print("\n📋 Custom Configuration")
        num_bugs = input("Number of bugs (Enter for all): ").strip()
        num_bugs = int(num_bugs) if num_bugs else None
        
        max_iter = input("Max iterations (default 5): ").strip()
        max_iter = int(max_iter) if max_iter else 5
        
        run_system(num_bugs=num_bugs, max_iter=max_iter)
        
    elif choice == '5':
        print("\n✅ System check complete!")
        print("\nTo run the system, choose options 1-3")
        
    elif choice == '6':
        print("\nExiting...")
        sys.exit(0)
    
    # Show results
    results_dir = Path('results')
    if results_dir.exists():
        result_files = list(results_dir.glob('*.json'))
        if result_files:
            latest = max(result_files, key=lambda p: p.stat().st_mtime)
            print(f"\n📊 Latest results: {latest}")
            print(f"\nView results with: cat {latest}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)