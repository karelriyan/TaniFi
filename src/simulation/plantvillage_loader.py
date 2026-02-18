
import os
import glob
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .image_filters import check_image_quality

class PlantVillageDataset(Dataset):
    """
    PlantVillage dataset loader for classification.
    Expects YOLO format labels (class_id x y w h) in a 'labels' directory
    and images in an 'images' directory.
    """
    def __init__(self, root, split='train', transform=None, filter_data=True):
        self.root = root
        self.split = split
        self.transform = transform
        self.filter_data = filter_data
        
        self.images_dir = os.path.join(root, 'images')
        self.labels_dir = os.path.join(root, 'labels')
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        # Sort for deterministic splitting
        self.image_files.sort()
        
        # Seeded shuffle and split
        random.seed(42)
        random.shuffle(self.image_files)
        
        num_total = len(self.image_files)
        num_train = int(0.8 * num_total)
        num_val = int(0.1 * num_total)
        # Remaining for test
        
        if split == 'train':
            self.image_files = self.image_files[:num_train]
        elif split == 'val':
            self.image_files = self.image_files[num_train:num_train+num_val]
        elif split == 'test':
            self.image_files = self.image_files[num_train+num_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
            
        print(f"PlantVillage split '{split}': {len(self.image_files)} images (from total {num_total})")

        # Optional: Load all labels into memory or verify them? 
        # For now, we'll verify and filter if requested during init to avoid runtime errors
        if self.filter_data:
            self.image_files = self._filter_images(self.image_files)
            print(f"After filtering: {len(self.image_files)} images")


        self.classes = self._load_classes()
        self.num_classes = len(self.classes)

    def _load_classes(self):
        # Attempt to load classes.yaml if it exists to verify num_classes
        classes_file = os.path.join(self.root, 'classes.yaml')
        if os.path.exists(classes_file):
            # Parse yaml manually to avoid pyyaml dependency if not desired, 
            # but usually yaml is available.
            pass
        return list(range(38)) # default 38 classes

    def _filter_images(self, image_files):
        valid_files = []
        skipped_quality = 0
        skipped_no_label = 0
        

        for i, p in enumerate(image_files):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(image_files)} images...", end='\r')
            # Check label existence
            base_name = os.path.basename(p)
            name_no_ext = os.path.splitext(base_name)[0]
            label_path = os.path.join(self.labels_dir, name_no_ext + '.txt')
            
            if not os.path.exists(label_path):
                skipped_no_label += 1
                continue
                
            # Check quality
            try:
                img = Image.open(p).convert('RGB')
                img_arr = np.array(img)
                if not check_image_quality(img_arr):
                    skipped_quality += 1
                    continue
                valid_files.append(p)
            except Exception as e:
                print(f"Error checking {p}: {e}")
                
        print(f"Filtered out {skipped_quality} low quality images and {skipped_no_label} missing labels.")
        return valid_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(base_name)[0]
        label_path = os.path.join(self.labels_dir, name_no_ext + '.txt')
        
        # Load Image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Load Label
        # Expecting class_id x y w h
        # We only take class_id for classification
        try:
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                else:
                    # Empty file? default to 0 or skip?
                    # Should have been filtered, but fail safe
                    class_id = 0 
        except Exception:
            class_id = 0
            
        return image, class_id
