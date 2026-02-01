"""
WeedsGalore Dataset Loader
Handles the date-based folder structure of WeedsGalore dataset
"""

import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class WeedsGaloreDataset(Dataset):
    """
    WeedsGalore Dataset with date-based folder structure

    Structure:
        data/raw/weedsgalore/weedsgalore-dataset/
        ├── 2023-05-25/
        │   ├── images/
        │   │   ├── {image_id}_R.png
        │   │   ├── {image_id}_G.png
        │   │   ├── {image_id}_B.png
        │   │   ├── {image_id}_NIR.png
        │   │   └── {image_id}_RE.png
        │   ├── instances/
        │   └── semantics/
        ├── 2023-05-30/
        ├── 2023-06-06/
        ├── 2023-06-15/
        └── splits/
            ├── train.txt
            ├── val.txt
            └── test.txt
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to weedsgalore-dataset/weedsgalore-dataset/
            split: 'train', 'val', or 'test'
            transform: Optional transform
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load split file
        split_file = self.root_dir / 'splits' / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # Get available date folders
        self.date_folders = sorted([
            d for d in self.root_dir.iterdir() 
            if d.is_dir() and d.name.startswith('2023-')
        ])
        
        print(f"Loaded {len(self.image_ids)} images for {split} split")
        print(f"Date folders: {[d.name for d in self.date_folders]}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def _find_image_folder(self, image_id):
        """Find image folder for given image_id"""
        # image_id format: "2023-05-25_0750" -> date is "2023-05-25"
        date_prefix = '-'.join(image_id.split('_')[0].split('-')[:3])

        for date_folder in self.date_folders:
            if date_folder.name == date_prefix:
                return date_folder / 'images'

        return None

    def _load_rgb_image(self, image_id):
        """
        Load RGB image from multispectral channels.

        WeedsGalore stores images as separate channels:
        - {image_id}_R.png (Red)
        - {image_id}_G.png (Green)
        - {image_id}_B.png (Blue)
        - {image_id}_NIR.png (Near-Infrared)
        - {image_id}_RE.png (Red Edge)

        This method combines R, G, B channels into an RGB image.
        """
        img_folder = self._find_image_folder(image_id)
        if img_folder is None:
            return None

        # Load R, G, B channels
        r_path = img_folder / f'{image_id}_R.png'
        g_path = img_folder / f'{image_id}_G.png'
        b_path = img_folder / f'{image_id}_B.png'

        # Check if all channels exist
        if not (r_path.exists() and g_path.exists() and b_path.exists()):
            # Fallback: try loading as single image (for compatibility)
            single_path = img_folder / f'{image_id}.png'
            if single_path.exists():
                return Image.open(single_path).convert('RGB')
            return None

        # Load individual channels as grayscale
        r_channel = np.array(Image.open(r_path).convert('L'))
        g_channel = np.array(Image.open(g_path).convert('L'))
        b_channel = np.array(Image.open(b_path).convert('L'))

        # Stack into RGB image
        rgb_array = np.stack([r_channel, g_channel, b_channel], axis=-1)

        return Image.fromarray(rgb_array, mode='RGB')
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load RGB image from multispectral channels
        image = self._load_rgb_image(image_id)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_id}")

        # For now, create dummy labels (0-9 classes)
        # TODO: Parse actual labels from instances/semantics
        label = np.random.randint(0, 10)

        if self.transform:
            image = self.transform(image)

        return image, label


def create_weedsgalore_loaders(batch_size=8):
    """Create train/val/test dataloaders"""
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # Path to dataset (relative to project root)
    project_root = Path(__file__).parent.parent.parent  # src/simulation -> src -> project root
    root = project_root / 'data/raw/weedsgalore/weedsgalore-dataset'
    
    # Simple transform for testing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Small for quick testing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = WeedsGaloreDataset(root, 'train', transform)
    val_dataset = WeedsGaloreDataset(root, 'val', transform)
    test_dataset = WeedsGaloreDataset(root, 'test', transform)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test loader
    print("Testing WeedsGalore loader...")
    
    train_loader, val_loader, test_loader = create_weedsgalore_loaders(batch_size=4)
    
    # Test one batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break
    
    print("✅ WeedsGalore loader works!")