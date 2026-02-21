# src/simulation/data.py

from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import FakeData
from .weedsgalore_loader import WeedsGaloreDataset
from .plantvillage_loader import PlantVillageDataset


class LabelOffsetDataset(Dataset):
    """Wraps a dataset and adds an offset to all labels.
    Safe for both Classification (int) and YOLO Object Detection (array) formats.
    """
    def __init__(self, dataset, label_offset, num_classes):
        self.dataset = dataset
        self.label_offset = label_offset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Safety check: if label is YOLO format [class_id, x, y, w, h],
        # extract only class_id (first element) and discard bounding box coords.
        if isinstance(label, (list, tuple, np.ndarray)):
            label = int(label[0]) if len(label) > 0 else 0
        elif isinstance(label, torch.Tensor) and label.numel() > 1:
            label = int(label[0].item())
        elif isinstance(label, torch.Tensor):
            label = int(label.item())
        else:
            label = int(label)

        return image, label + self.label_offset


def create_combined_dataset(img_size=224, split='train'):
    """Create combined WeedsGalore + PlantVillage dataset.
    
    Label mapping:
        WeedsGalore classes 0-2  → labels 0-2   (3 weed types)
        PlantVillage classes 0-37 → labels 3-40  (38 plant diseases)
        Total: 41 classes
    """
    wg = create_weedsgalore_dataset(img_size=img_size, split=split)   # 3 classes (0-2)
    pv = create_plantvillage_dataset(img_size=img_size, split=split)  # 38 classes (0-37)

    # Offset PlantVillage labels by 3 so they become 3-40
    pv_offset = LabelOffsetDataset(pv, label_offset=3, num_classes=41)

    combined = ConcatDataset([wg, pv_offset])
    combined.num_classes = 41

    wg_len = len(wg)
    pv_len = len(pv)
    print(f"Combined dataset ({split}): {wg_len} WeedsGalore + {pv_len} PlantVillage = {wg_len + pv_len} total, 41 classes")

    return combined


def create_weedsgalore_dataset(img_size=64, split='train'):
    """Create WeedsGalore dataset with real labels from semantic masks."""
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    project_root = Path(__file__).parent.parent.parent
    dataset_root = project_root / 'data/weedsgalore/weedsgalore-dataset'
    if not dataset_root.exists():
        raise FileNotFoundError(f"WeedsGalore dataset not found at: {dataset_root}")
    return WeedsGaloreDataset(root_dir=dataset_root, split=split, transform=transform)


def create_plantvillage_dataset(img_size=224, split='train'):
    """Create PlantVillage dataset."""
    # Standard Imagenet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
        
    project_root = Path(__file__).parent.parent.parent
    dataset_root = project_root / 'data/archive/PlantVillage_for_object_detection/Dataset'
    if not dataset_root.exists():
         raise FileNotFoundError(f"PlantVillage dataset not found at: {dataset_root}")
         
    return PlantVillageDataset(root=dataset_root, split=split, transform=transform)


def create_dataset(dataset_name='synthetic', use_real_data=None, num_samples=10000,
                   img_size=64, num_classes=3, split='train'):
    """
    Create dataset – supports 'synthetic', 'weedsgalore', 'plantvillage', 'combined'.
    
    Args:
        dataset_name (str): Name of the dataset ('synthetic', 'weedsgalore', 'plantvillage', 'combined').
        use_real_data (bool, optional): Legacy argument. If True, defaults to 'weedsgalore'.
        num_samples (int): Number of samples for synthetic data.
        img_size (int): Image size.
        num_classes (int): Number of classes for synthetic data.
        split (str): 'train', 'val', or 'test'.
    """
    # Backward compatibility logic
    if use_real_data is not None:
        if use_real_data:
            dataset_name = 'weedsgalore'
        else:
            dataset_name = 'synthetic'
            
    if dataset_name == 'weedsgalore':
        return create_weedsgalore_dataset(img_size=img_size, split=split)
    elif dataset_name == 'plantvillage':
        return create_plantvillage_dataset(img_size=img_size, split=split)
    elif dataset_name == 'combined':
        return create_combined_dataset(img_size=img_size, split=split)
    else:
        # Synthetic
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return FakeData(size=num_samples,
                        image_size=(3, img_size, img_size),
                        num_classes=num_classes,
                        transform=transform)

__all__ = ["create_dataset", "create_weedsgalore_dataset", "create_plantvillage_dataset", "create_combined_dataset", "LabelOffsetDataset"]

