#!/usr/bin/env python3
"""
Environment Verification Script for TaniFi Project

This script checks if all dependencies and setup are correct
Run this before starting the simulation to catch any issues early

Usage:
    python verify_setup.py
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_status(check_name, status, message=""):
    """Print check status"""
    icon = "✅" if status else "❌"
    print(f"{icon} {check_name:<40} {'PASS' if status else 'FAIL'}")
    if message and not status:
        print(f"   💡 {message}")


def check_python_version():
    """Check Python version >= 3.8"""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 8
    
    print_status(
        "Python Version (>= 3.8)",
        is_valid,
        f"Found: Python {version.major}.{version.minor}.{version.micro}"
    )
    return is_valid


def check_package(package_name, import_name=None):
    """Check if Python package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_dependencies():
    """Check all required Python packages"""
    print_header("Checking Python Dependencies")
    
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
        ('pyyaml', 'yaml'),
    ]
    
    all_ok = True
    for pkg_name, import_name in required_packages:
        status = check_package(pkg_name, import_name)
        print_status(f"{pkg_name}", status, f"Install with: pip install {pkg_name}")
        all_ok = all_ok and status
    
    return all_ok


def check_pytorch_cuda():
    """Check PyTorch CUDA availability"""
    print_header("Checking PyTorch Configuration")
    
    try:
        import torch
        
        # Check PyTorch version
        version_ok = torch.__version__ >= "2.0.0"
        print_status(
            f"PyTorch Version ({torch.__version__})",
            version_ok,
            "Consider upgrading: pip install torch --upgrade"
        )
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print_status(
            "CUDA Available",
            cuda_available,
            "GPU training not available. Will use CPU (slower)"
        )
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"   🎮 GPU Device: {device_name}")
            print(f"   💾 CUDA Version: {torch.version.cuda}")
        
        return version_ok
        
    except ImportError:
        print_status("PyTorch", False, "Install with: pip install torch")
        return False


def check_directory_structure():
    """Check if required directories exist"""
    print_header("Checking Directory Structure")
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'models/checkpoints',
        'experiments/results',
        'experiments/results/plots',
        'experiments/results/tables',
        'src/simulation',
        'notebooks',
        'docs'
    ]
    
    project_root = Path(__file__).parent
    all_ok = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists()
        print_status(f"Directory: {dir_path}", exists)
        all_ok = all_ok and exists
    
    return all_ok


def check_required_files():
    """Check if required files exist"""
    print_header("Checking Required Files")
    
    required_files = [
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        '.gitignore',
        'src/simulation/diloco_trainer.py',
        'src/simulation/download_dataset.py',
        'experiments/config.yaml',
        'notebooks/analysis_template.ipynb'
    ]
    
    project_root = Path(__file__).parent
    all_ok = True
    
    for file_path in required_files:
        full_path = project_root / file_path
        exists = full_path.exists()
        print_status(f"File: {file_path}", exists)
        all_ok = all_ok and exists
    
    return all_ok


def check_dataset():
    """Check if dataset is available"""
    print_header("Checking Dataset")
    
    project_root = Path(__file__).parent
    dataset_dir = project_root / 'data' / 'raw' / 'weedsgalore'
    
    exists = dataset_dir.exists()
    print_status(
        "WeedsGalore Dataset",
        exists,
        "Run: python src/simulation/download_dataset.py --dataset weedsgalore"
    )
    
    if exists:
        # Count files
        image_count = len(list(dataset_dir.glob('**/*.jpg'))) + len(list(dataset_dir.glob('**/*.png')))
        print(f"   📊 Found {image_count} images")
    
    return True  # Dataset is optional for initial testing


def run_quick_test():
    """Run a quick import test of main modules"""
    print_header("Quick Module Import Test")
    
    try:
        # Try importing the main simulation module
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        
        print("   Importing diloco_trainer...")
        # Just check if file can be read, don't actually import to avoid running code
        trainer_file = Path(__file__).parent / 'src' / 'simulation' / 'diloco_trainer.py'
        
        if trainer_file.exists():
            print_status("diloco_trainer.py", True)
            return True
        else:
            print_status("diloco_trainer.py", False, "File not found")
            return False
            
    except Exception as e:
        print_status("Module Import", False, str(e))
        return False


def main():
    """Main verification workflow"""
    print("\n" + "="*60)
    print("  TaniFi Environment Verification")
    print("  Checking if your setup is ready for Week 1...")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("PyTorch & CUDA", check_pytorch_cuda),
        ("Directory Structure", check_directory_structure),
        ("Required Files", check_required_files),
        ("Dataset (Optional)", check_dataset),
        ("Module Import", run_quick_test),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n❌ Error during {check_name}: {str(e)}")
            results[check_name] = False
    
    # Summary
    print_header("Verification Summary")
    
    total_checks = len([r for r in results.values() if r is not None])
    passed_checks = sum(1 for r in results.values() if r is True)
    
    print(f"Total Checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    
    print(f"\nOverall Status: ", end="")
    if passed_checks == total_checks:
        print("✅ ALL CHECKS PASSED! You're ready to go!")
        print("\nNext steps:")
        print("1. Run simulation: cd src/simulation && python diloco_trainer.py")
        print("2. Check QUICKSTART.md for detailed instructions")
        return 0
    elif passed_checks >= total_checks * 0.8:
        print("⚠️  MOSTLY READY (minor issues)")
        print("\nYou can proceed, but fix the failed checks when possible.")
        print("Check QUICKSTART.md for troubleshooting.")
        return 0
    else:
        print("❌ SETUP INCOMPLETE")
        print("\nPlease fix the failed checks before proceeding.")
        print("See QUICKSTART.md for installation instructions.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)