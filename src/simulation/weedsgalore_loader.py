"""
WeedsGalore Dataset Loader
Handles the date-based folder structure of WeedsGalore dataset.

Label derivation: Each image is classified by its dominant (most frequent)
non-background semantic class from the segmentation masks.

Semantic mask classes:
    0 = background (soil)
    1 = weed type 1 (most common)
    2 = weed type 2
    3 = weed type 3
    4 = weed type 4 (rare)
    5 = crop

Output labels (mapped for classification):
    0 = dominant class 1 (weed type 1)
    1 = dominant class 2 (weed type 2)
    2 = dominant class 3 (weed type 3)

NUM_CLASSES = 3
"""

import os
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from .image_filters import check_image_quality

NUM_CLASSES = 3

# Mapping from semantic mask class to output label index
SEMANTIC_TO_LABEL = {1: 0, 2: 1, 3: 2}


class WeedsGaloreDataset(Dataset):
    """
    WeedsGalore Dataset with date-based folder structure.

    Labels are derived from semantic segmentation masks using the
    dominant non-background class strategy.

    Structure:
        data/weedsgalore/weedsgalore-dataset/
        ├── 2023-05-25/
        │   ├── images/
        │   │   ├── {image_id}_R.png
        │   │   ├── {image_id}_G.png
        │   │   ├── {image_id}_B.png
        │   │   ├── {image_id}_NIR.png
        │   │   └── {image_id}_RE.png
        │   ├── instances/
        │   └── semantics/
        │       └── {image_id}.png  (pixel values 0-5)
        ├── 2023-05-30/
        ├── 2023-06-06/
        ├── 2023-06-15/
        └── splits/
            ├── train.txt
            ├── val.txt
            └── test.txt
    """

    def __init__(self, root_dir, split='train', transform=None, filter_data=True):
        """
        Args:
            root_dir: Path to weedsgalore-dataset/
            split: 'train', 'val', or 'test'
            transform: Optional transform
            filter_data: Whether to apply quality filtering on images.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.filter_data = filter_data

        # Load split file
        split_file = self.root_dir / 'splits' / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines() if line.strip()]

        # Get available date folders
        self.date_folders = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and d.name.startswith('2023-')
        ])

        # Statistics for filtering
        self.accepted_count = 0
        self.rejected_count = 0

        # If filtering is enabled, validate each image using the quality filter
        if self.filter_data:
            valid_ids = []
            valid_labels = []
            # Compute labels once for all images
            all_labels = self._compute_labels()
            for img_id, label in zip(self.image_ids, all_labels):
                pil_img = self._load_rgb_image(img_id)
                if pil_img is None:
                    self.rejected_count += 1
                    continue
                
                # Convert PIL (RGB) to OpenCV (BGR) for checking
                img_bgr = np.array(pil_img)[:, :, ::-1]
                is_valid, _ = check_image_quality(img_bgr)
                
                if is_valid:
                    valid_ids.append(img_id)
                    valid_labels.append(label)
                    self.accepted_count += 1
                else:
                    self.rejected_count += 1
            self.image_ids = valid_ids
            self.labels = valid_labels
        else:
            # No filtering; compute labels once
            self.labels = self._compute_labels()

        self.num_classes = NUM_CLASSES

        label_dist = Counter(self.labels)
        print(f"[{split.upper()}] {len(self.image_ids)} images after QC, "
              f"accepted: {self.accepted_count}, rejected: {self.rejected_count}, "
              f"label distribution: {dict(sorted(label_dist.items()))}")

    def _find_date_folder(self, image_id):
        """Find the date folder for a given image_id."""
        date_prefix = '-'.join(image_id.split('_')[0].split('-')[:3])
        for date_folder in self.date_folders:
            if date_folder.name == date_prefix:
                return date_folder
        return None

    def _load_semantic_mask(self, image_id):
        """Load semantic segmentation mask for a given image_id."""
        date_folder = self._find_date_folder(image_id)
        if date_folder is None:
            raise FileNotFoundError(f"Date folder not found for: {image_id}")

        mask_path = date_folder / 'semantics' / f'{image_id}.png'
        if not mask_path.exists():
            raise FileNotFoundError(f"Semantic mask not found: {mask_path}")

        return np.array(Image.open(mask_path))

    def _compute_labels(self):
        """Derive classification labels from semantic segmentation masks.

        Strategy: dominant (most frequent) non-background class per image.
        Maps semantic classes {1,2,3,4,5} to label indices {0,1,2}.
        Classes 4 and 5 are rare and mapped to label 2 (grouped with class 3).
        """
        labels = []
        for image_id in self.image_ids:
            try:
                mask = self._load_semantic_mask(image_id)
                non_bg = mask[mask > 0]
                if len(non_bg) > 0:
                    values, counts = np.unique(non_bg, return_counts=True)
                    dominant = int(values[np.argmax(counts)])
                    label = SEMANTIC_TO_LABEL.get(dominant, 2)  # rare classes -> 2
                else:
                    label = 0  # fallback: all background
            except FileNotFoundError:
                label = 0  # fallback
            labels.append(label)
        return labels

    def _find_image_folder(self, image_id):
        """Find image folder for given image_id."""
        date_folder = self._find_date_folder(image_id)
        if date_folder is not None:
            return date_folder / 'images'
        return None

    def _find_image_path(self, image_id):
        """Return the filesystem path to the RGB image for *image_id*.

        Handles both combined image files and separate channel files.
        """
        img_folder = self._find_image_folder(image_id)
        if img_folder is None:
            return None
        # Prefer combined image if exists
        combined = img_folder / f"{image_id}.png"
        if combined.exists():
            return combined
        # Otherwise check for separate channel files
        r_path = img_folder / f"{image_id}_R.png"
        g_path = img_folder / f"{image_id}_G.png"
        b_path = img_folder / f"{image_id}_B.png"
        if r_path.exists() and g_path.exists() and b_path.exists():
            return r_path
        return None

    def _load_rgb_image(self, image_id):
        """
        Load RGB image from multispectral channels.

        WeedsGalore stores images as separate channels:
        - {image_id}_R.png (Red)
        - {image_id}_G.png (Green)
        - {image_id}_B.png (Blue)

        This method combines R, G, B channels into an RGB image.
        """
        img_folder = self._find_image_folder(image_id)
        if img_folder is None:
            return None

        r_path = img_folder / f'{image_id}_R.png'
        g_path = img_folder / f'{image_id}_G.png'
        b_path = img_folder / f'{image_id}_B.png'

        if not (r_path.exists() and g_path.exists() and b_path.exists()):
            single_path = img_folder / f'{image_id}.png'
            if single_path.exists():
                return Image.open(single_path).convert('RGB')
            return None

        r_img = Image.open(r_path)
        def load_channel(path):
            img = Image.open(path)
            if img.mode == 'I;16':
                arr = np.array(img)
                # Scale 16-bit (0-65535) to 8-bit (0-255)
                return (arr / 257).astype(np.uint8)
            return np.array(img.convert('L'))

        r_channel = load_channel(r_path)
        g_channel = load_channel(g_path)
        b_channel = load_channel(b_path)

        rgb_array = np.stack([r_channel, g_channel, b_channel], axis=-1)
        return Image.fromarray(rgb_array, mode='RGB')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image = self._load_rgb_image(image_id)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_id}")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_weedsgalore_loaders(batch_size=8, img_size=224, oversample_minority=False):
    """Create train/val/test dataloaders with optional oversampling for minority classes.

    Args:
        batch_size (int): Batch size for all loaders.
        img_size (int): Target image size (square).
        oversample_minority (bool): If True, applies weighted random sampling to the training
            set to mitigate class imbalance.
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import torch
    import numpy as np

    project_root = Path(__file__).parent.parent.parent
    root = project_root / 'data/weedsgalore/weedsgalore-dataset'

    # Strong augmentations for training to improve robustness and handle class imbalance
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Val/test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = WeedsGaloreDataset(root, 'train', train_transform, filter_data=True)
    val_dataset = WeedsGaloreDataset(root, 'val', eval_transform, filter_data=True)
    test_dataset = WeedsGaloreDataset(root, 'test', eval_transform, filter_data=True)

    if oversample_minority:
        class_counts = np.bincount(train_dataset.labels, minlength=NUM_CLASSES)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[label] for label in train_dataset.labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("Testing WeedsGalore loader with real labels...")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_weedsgalore_loaders(batch_size=4)

    print(f"\nTrain: {len(train_loader.dataset)} samples")
    print(f"Val:   {len(val_loader.dataset)} samples")
    print(f"Test:  {len(test_loader.dataset)} samples")

    for images, labels in train_loader:
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels: {labels}")
        break

    print("\n✅ WeedsGalore loader works with real labels!")