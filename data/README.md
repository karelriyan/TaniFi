# Data Structure

This folder stores all datasets used for the TaniFi simulation. No data is preprocessed and stored — all transformations are applied *on-the-fly* during training.

---

## Supported Datasets

| Dataset      | Classes               | Image Size | Scenario             |
|--------------|-----------------------|------------|----------------------|
| weedsgalore  | 3 (weed types)        | 224×224    | Weed detection       |
| plantvillage | 38 (plant diseases)   | 224×224    | Disease classification |
| synthetic    | 3                     | 64×64      | Quick testing        |

---

## 📁 Folder Structure

```
data/
├── weedsgalore/                        # WeedsGalore dataset
│   └── weedsgalore-dataset/
│       ├── 2023-05-25/
│       │   ├── images/                 # Multispectral images
│       │   │   ├── {image_id}_R.png
│       │   │   ├── {image_id}_G.png
│       │   │   ├── {image_id}_B.png
│       │   │   ├── {image_id}_NIR.png
│       │   │   └── {image_id}_RE.png
│       │   ├── instances/              # Instance masks
│       │   └── semantics/              # Semantic masks (classes 0-5)
│       ├── 2023-05-30/
│       ├── 2023-06-06/
│       ├── 2023-06-15/
│       └── splits/
│           ├── train.txt
│           ├── val.txt
│           └── test.txt
├── plantvillage/                       # PlantVillage dataset (YOLO format)
│   ├── images/                         # Image files (.jpg/.png)
│   │   ├── Apple___Apple_scab_001.jpg
│   │   └── ...
│   ├── labels/                         # YOLO label files (.txt per image)
│   │   ├── Apple___Apple_scab_001.txt  # Format: class_id cx cy w h
│   │   └── ...
│   └── classes.yaml                    # List of 38 class names (optional)
└── README.md                           # This file
```

---

## 🌿 WeedsGalore Dataset

- **Source**: WeedsGalore multispectral agricultural dataset
- **Labels**: Derived from semantic segmentation masks (dominant non-background class)
- **Classes**: 3 weed types (weed type 1, 2, 3+)
- **Imbalance**: 74% class 0, 20% class 1, 6% class 2
- **Split**: 80% train / 10% val / 10% test (via `splits/`)

### Download
```bash
python src/simulation/download_dataset.py --dataset weedsgalore

# Or download a small sample for testing
python src/simulation/download_dataset.py --dataset weedsgalore --sample-size 100
```

### Transformations (on-the-fly)
1. RGB channel extraction (from multispectral images)
2. Resize to 224×224
3. Data augmentation (random flips, rotations, color jitter)
4. Normalization (ImageNet statistics)

---

## 🌱 PlantVillage Dataset

- **Source**: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) — 54,000+ plant leaf images
- **Format**: YOLO (`.txt` label per image: `class_id cx cy w h`)
- **Classes**: 38 classes (26 diseases + 12 healthy conditions across various crops)
- **Split**: 80% train / 10% val / 10% test (deterministic shuffle, seed=42)
- **Filtering**: Automatically skips blurry images or images with missing label files

### Example Classes
```
0: Apple___Apple_scab
1: Apple___Black_rot
2: Apple___Cedar_apple_rust
3: Apple___healthy
...
37: Tomato___Tomato_mosaic_virus
```

### YOLO Label Format
```
# File: labels/Apple___Apple_scab_001.txt
0 0.5 0.5 1.0 1.0
# Format: class_id center_x center_y width height (all values relative 0-1)
```

### Download & Setup
```bash
# From Kaggle (requires kaggle CLI)
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d data/plantvillage/

# Ensure structure: data/plantvillage/images/ and data/plantvillage/labels/
```

### Experiment Configuration
To use PlantVillage, set in the YAML config:
```yaml
dataset:
  name: plantvillage
  image_size: 224
```

Or via command line:
```bash
./venv/bin/python experiments/run_experiments.py --dataset plantvillage
```

### Transformations (on-the-fly)
1. Load RGB image from `images/`
2. Read `class_id` from `.txt` file in `labels/`
3. Quality filtering (blur detection via Laplacian, threshold ≥ 100)
4. Resize to target `img_size` (default 224×224)
5. Normalization (ImageNet statistics)

---

> **Note**: There is no `processed/` folder — all transformations are applied in-memory during training. This saves storage space and allows flexible augmentation strategies.