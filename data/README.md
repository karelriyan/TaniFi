# Data Structure

## WeedsGalore Dataset

This folder contains the WeedsGalore agricultural dataset for weed classification.

### Structure
```
data/
├── weedsgalore/                    # WeedsGalore dataset
│   └── weedsgalore-dataset/        # Downloaded dataset
│       ├── 2023-05-25/             # Date folder 1
│       │   ├── images/             # Multispectral channels
│       │   │   ├── {image_id}_R.png
│       │   │   ├── {image_id}_G.png
│       │   │   ├── {image_id}_B.png
│       │   │   ├── {image_id}_NIR.png
│       │   │   └── {image_id}_RE.png
│       │   ├── instances/          # Instance masks
│       │   └── semantics/          # Semantic masks (0-5 classes)
│       ├── 2023-05-30/             # Date folder 2
│       ├── 2023-06-06/             # Date folder 3
│       ├── 2023-06-15/             # Date folder 4
│       └── splits/                 # Predefined splits
│           ├── train.txt           # Training image IDs
│           ├── val.txt             # Validation image IDs
│           └── test.txt            # Test image IDs
└── README.md                       # This file
```

### Dataset Details
- **Original Source**: WeedsGalore multispectral agricultural dataset
- **Label Derivation**: From semantic segmentation masks using dominant non-background class
- **Classes**: 3 weed types (weed type 1, 2, 3+)
- **Image Size**: Original 600×600, resized to 224×224 for training

### Download Instructions
```bash
# Download dataset
python src/simulation/download_dataset.py --dataset weedsgalore

# Or download sample for testing
python src/simulation/download_dataset.py --dataset weedsgalore --sample-size 100
```

### Processing
All data processing is done **on-the-fly** during training by `WeedsGaloreDataset` class. No pre-processed data is stored.

Key transformations applied during training:
1. RGB channel extraction (R, G, B channels only)
2. Image resizing (224×224)
3. Data augmentation (random flips, rotations, color jitter)
4. Normalization (ImageNet statistics)

### Notes
- No `processed/` folder exists - all transformations are applied in-memory during training
- This approach saves storage space and allows flexible augmentation strategies
- The dataset is imbalanced (74% class 0, 20% class 1, 6% class 2)