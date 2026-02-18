# src/simulation/data.py

from pathlib import Path
from torchvision import transforms
from torchvision.datasets import FakeData
from .weedsgalore_loader import WeedsGaloreDataset
from .plantvillage_loader import PlantVillageDataset

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
    Create dataset – supports 'synthetic', 'weedsgalore', 'plantvillage'.
    
    Args:
        dataset_name (str): Name of the dataset ('synthetic', 'weedsgalore', 'plantvillage').
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

__all__ = ["create_dataset", "create_weedsgalore_dataset", "create_plantvillage_dataset"]

