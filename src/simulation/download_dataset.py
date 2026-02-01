"""
Dataset Download Script for TaniFi Federated Learning

This script automates the download of agricultural datasets for the research project.
Supports WeedsGalore, Global Wheat, and other datasets.

Usage:
    python download_dataset.py --dataset weedsgalore --output ../data/raw
    python download_dataset.py --dataset global-wheat --sample-size 1000
"""

import os
import argparse
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import gdown


class DatasetDownloader:
    """Handle dataset downloads from various sources"""
    
    DATASETS = {
        'weedsgalore': {
            'kaggle': 'vinayakshanawad/weedsgalore',
            'gdrive': None,  # Add Google Drive link if available
            'size': '2.5GB'
        },
        'global-wheat': {
            'kaggle': 'c/global-wheat-detection',
            'gdrive': None,
            'size': '1.2GB'
        },
        'plantvillage': {
            'kaggle': 'emmarex/plantdisease',
            'gdrive': None,
            'size': '2.1GB'
        }
    }
    
    def __init__(self, dataset_name, output_dir='../data/raw', sample_size=None):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.sample_size = sample_size
        
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not supported. "
                           f"Choose from: {list(self.DATASETS.keys())}")
        
        self.dataset_info = self.DATASETS[dataset_name]
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_from_kaggle(self):
        """Download dataset from Kaggle using kaggle API"""
        try:
            import kaggle
            
            print(f"📥 Downloading {self.dataset_name} from Kaggle...")
            print(f"   Estimated size: {self.dataset_info['size']}")
            
            # Download dataset
            kaggle_path = self.dataset_info['kaggle']
            
            if kaggle_path.startswith('c/'):
                # Competition dataset
                competition_name = kaggle_path.replace('c/', '')
                kaggle.api.competition_download_files(
                    competition_name,
                    path=str(self.output_dir / self.dataset_name)
                )
            else:
                # Regular dataset
                kaggle.api.dataset_download_files(
                    kaggle_path,
                    path=str(self.output_dir / self.dataset_name),
                    unzip=True
                )
            
            print(f"✅ Download complete! Saved to: {self.output_dir / self.dataset_name}")
            return True
            
        except Exception as e:
            print(f"❌ Kaggle download failed: {str(e)}")
            print("\n💡 Troubleshooting:")
            print("   1. Install kaggle: pip install kaggle")
            print("   2. Set up API credentials:")
            print("      - Go to kaggle.com/account")
            print("      - Create API token")
            print("      - Place kaggle.json in ~/.kaggle/")
            print("   3. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
            return False
    
    def download_from_gdrive(self):
        """Download dataset from Google Drive"""
        gdrive_id = self.dataset_info.get('gdrive')
        
        if not gdrive_id:
            print(f"❌ No Google Drive link available for {self.dataset_name}")
            return False
        
        try:
            print(f"📥 Downloading {self.dataset_name} from Google Drive...")
            output_path = self.output_dir / self.dataset_name / 'dataset.zip'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            gdown.download(id=gdrive_id, output=str(output_path), quiet=False)
            
            # Extract zip
            print("📦 Extracting files...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(output_path.parent)
            
            # Clean up zip file
            output_path.unlink()
            
            print(f"✅ Download complete! Saved to: {output_path.parent}")
            return True
            
        except Exception as e:
            print(f"❌ Google Drive download failed: {str(e)}")
            return False
    
    def download_sample_dataset(self):
        """Download a small sample dataset for quick testing"""
        print(f"📥 Creating sample dataset (size: {self.sample_size or 100} images)...")
        
        # This would typically download a subset
        # For now, create placeholder structure
        sample_dir = self.output_dir / f"{self.dataset_name}_sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        (sample_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (sample_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (sample_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (sample_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Sample dataset structure created at: {sample_dir}")
        print(f"   Next step: Add your sample images to the created directories")
        return True
    
    def verify_download(self):
        """Verify that dataset was downloaded correctly"""
        dataset_path = self.output_dir / self.dataset_name
        
        if not dataset_path.exists():
            print(f"❌ Dataset directory not found: {dataset_path}")
            return False
        
        # Check for expected structure
        required_dirs = ['images', 'labels']
        missing_dirs = [d for d in required_dirs 
                       if not (dataset_path / d).exists()]
        
        if missing_dirs:
            print(f"⚠️  Warning: Missing directories: {missing_dirs}")
            print(f"   Dataset may need manual organization")
            return False
        
        print(f"✅ Dataset verification passed!")
        return True
    
    def run(self, method='kaggle'):
        """Main download orchestration"""
        print(f"\n{'='*60}")
        print(f"TaniFi Dataset Downloader")
        print(f"{'='*60}\n")
        
        # Create output directory
        print(f"📁 Output directory: {self.output_dir}")
        
        # Download based on method
        success = False
        
        if self.sample_size:
            success = self.download_sample_dataset()
        elif method == 'kaggle':
            success = self.download_from_kaggle()
        elif method == 'gdrive':
            success = self.download_from_gdrive()
        else:
            print(f"❌ Unknown download method: {method}")
            return
        
        # Verify if real download was attempted
        if success and not self.sample_size:
            self.verify_download()
        
        print(f"\n{'='*60}")
        print("Next steps:")
        print(f"1. Verify data structure in: {self.output_dir / self.dataset_name}")
        print(f"2. Run preprocessing: python preprocess_data.py")
        print(f"3. Create federated splits: python data_partitioner.py")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download agricultural datasets for TaniFi research'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['weedsgalore', 'global-wheat', 'plantvillage'],
        help='Dataset name to download'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../data/raw',
        help='Output directory for downloaded data'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='kaggle',
        choices=['kaggle', 'gdrive'],
        help='Download method (kaggle or gdrive)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Create sample dataset with N images instead of full download'
    )
    
    args = parser.parse_args()
    
    # Initialize and run downloader
    downloader = DatasetDownloader(
        dataset_name=args.dataset,
        output_dir=args.output,
        sample_size=args.sample_size
    )
    
    downloader.run(method=args.method)


if __name__ == '__main__':
    main()