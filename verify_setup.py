#!/usr/bin/env python3
"""
Environment Verification Script for TaniFi Project
Checks if all dependencies, datasets, and directories are properly set up.
"""

import sys
import subprocess
import importlib
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"{text:^60}")
    print("=" * 60)


def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version OK (>= 3.8)")
        return True
    else:
        print("❌ Python 3.8 or higher required")
        return False


def check_required_packages():
    """Check if required packages are installed"""
    print_header("Checking Required Packages")
    
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'tqdm',
        'pyyaml',
        'pillow',
        'ultralytics',
        'pathlib',
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            all_installed = False
    
    return all_installed


def check_directory_structure():
    """Check if required directories exist"""
    print_header("Checking Directory Structure")
    
    required_dirs = [
        'data/weedsgalore',
        'models/checkpoints',
        'experiments/results',
        'experiments/results/plots',
        'experiments/results/tables',
        'src/simulation',
        'notebooks',
        'docs'
    ]
    
    project_root = Path(__file__).parent
    all_exist = True
    
    for dir_path in required_dirs:
        dir_full = project_root / dir_path
        if dir_full.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def check_dataset():
    """Check if WeedsGalore dataset is available"""
    print_header("Checking WeedsGalore Dataset")
    
    project_root = Path(__file__).parent
    dataset_dir = project_root / 'data' / 'weedsgalore' / 'weedsgalore-dataset'
    
    if not dataset_dir.exists():
        print(f"❌ Dataset not found at: {dataset_dir}")
        print("   Run: python src/simulation/download_dataset.py --dataset weedsgalore")
        return False
    
    # Check for required subdirectories
    required = ['splits']
    split_files = ['train.txt', 'val.txt', 'test.txt']
    
    all_ok = True
    for req in required:
        req_path = dataset_dir / req
        if req_path.exists():
            print(f"✅ {req}/ directory exists")
            
            # Check for split files
            for split_file in split_files:
                if (req_path / split_file).exists():
                    print(f"   ✅ {split_file}")
                else:
                    print(f"   ❌ Missing {split_file}")
                    all_ok = False
        else:
            print(f"❌ Missing {req}/ directory")
            all_ok = False
    
    # Check for at least one date folder
    date_folders = [d for d in dataset_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('2023-')]
    
    if date_folders:
        print(f"✅ Found {len(date_folders)} date folders")
        for folder in date_folders[:3]:  # Show first 3
            print(f"   - {folder.name}")
        if len(date_folders) > 3:
            print(f"   ... and {len(date_folders) - 3} more")
    else:
        print("❌ No date folders found (should have 2023-05-25, etc.)")
        all_ok = False
    
    return all_ok


def check_model_files():
    """Check if required model files exist"""
    print_header("Checking Model Files")
    
    project_root = Path(__file__).parent
    yolo_model = project_root / 'yolo11n-cls.pt'
    
    if yolo_model.exists():
        print(f"✅ YOLOv11 classification model found")
        size_mb = yolo_model.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ YOLOv11 model not found at: {yolo_model}")
        print("   Download with: wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n-cls.pt")
        return False


def check_gpu_availability():
    """Check if CUDA is available"""
    print_header("Checking GPU Availability")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU available: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
            return False
    except ImportError:
        print("❌ torch not installed")
        return False


def test_imports():
    """Test if key imports work"""
    print_header("Testing Imports")
    
    imports_to_test = [
        ('src.simulation.weedsgalore_loader', 'WeedsGaloreDataset'),
        ('src.simulation.diloco_trainer', 'DiLoCoCoordinator'),
    ]
    
    all_ok = True
    for module_path, obj_name in imports_to_test:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, obj_name):
                print(f"✅ {module_path}.{obj_name}")
            else:
                print(f"❌ {obj_name} not found in {module_path}")
                all_ok = False
        except ImportError as e:
            print(f"❌ Failed to import {module_path}: {e}")
            all_ok = False
    
    return all_ok


def run_quick_test():
    """Run a quick test of the data loader"""
    print_header("Running Quick Test")
    
    try:
        from src.simulation.weedsgalore_loader import create_weedsgalore_loaders
        
        print("Testing WeedsGalore data loader...")
        train_loader, val_loader, test_loader = create_weedsgalore_loaders(batch_size=2)
        
        print(f"✅ Train loader created: {len(train_loader.dataset)} samples")
        print(f"✅ Val loader created: {len(val_loader.dataset)} samples")
        print(f"✅ Test loader created: {len(test_loader.dataset)} samples")
        
        # Try to get one batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx == 0:
                print(f"✅ Batch shape: {images.shape}")
                print(f"✅ Labels: {labels.tolist()}")
                print(f"✅ Data types: images={images.dtype}, labels={labels.dtype}")
                break
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main verification function"""
    print("\n" + "=" * 60)
    print("TaniFi Environment Verification")
    print("=" * 60)
    
    checks = [
        ('Python Version', check_python_version),
        ('Required Packages', check_required_packages),
        ('Directory Structure', check_directory_structure),
        ('WeedsGalore Dataset', check_dataset),
        ('Model Files', check_model_files),
        ('GPU Availability', check_gpu_availability),
        ('Module Imports', test_imports),
        ('Quick Loader Test', run_quick_test),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    print_header("Verification Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\n{'=' * 60}")
    print(f"Overall: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("✅ All checks passed! You're ready to start experiments.")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nSuggested fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Download dataset: python src/simulation/download_dataset.py --dataset weedsgalore")
        print("3. Download model: wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n-cls.pt")
        print("4. Create missing directories manually")
        return 1


if __name__ == "__main__":
    sys.exit(main())