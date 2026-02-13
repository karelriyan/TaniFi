# Dataset Setup Guide

## WeedsGalore Dataset

### Overview
WeedsGalore is a comprehensive agricultural dataset containing images of crops and weeds, ideal for training object detection models in agricultural contexts.

### Download Options

#### Option 1: Manual Download (Recommended for first time)
1. Visit the WeedsGalore dataset page:
   - Kaggle: https://www.kaggle.com/datasets/vinayakshanawad/weedsgalore
   - Or alternative sources from research papers

2. Download the dataset ZIP file

3. Extract to `data/raw/weedsgalore/`

#### Option 2: Automated Download (Using script)
```bash
# Make sure you have Kaggle API credentials set up
# Create ~/.kaggle/kaggle.json with your API credentials

python src/simulation/download_dataset.py --dataset weedsgalore
```

#### Option 3: Alternative - Global Wheat Dataset
If WeedsGalore is unavailable, use Global Wheat Head Detection Dataset:
```bash
python src/simulation/download_dataset.py --dataset global-wheat
```

### Dataset Structure (Expected)
```
data/raw/weedsgalore/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Preprocessing
After download, run the preprocessing script:
```bash
python src/simulation/preprocess_data.py --input data/raw/weedsgalore --output data/processed/weedsgalore
```

This will:
- Resize images to standard dimensions (640x640 for YOLOv11)
- Normalize annotations
- Create train/val/test splits (80/10/10)
- Generate data manifest files

### Data Statistics (Target)
- **Training samples**: ~8,000 images
- **Validation samples**: ~1,000 images
- **Test samples**: ~1,000 images
- **Classes**: Multiple crop disease and weed categories
- **Image format**: JPEG/PNG
- **Annotation format**: YOLO format (txt files)

### Simulating "Farmer Shards"
For federated learning simulation, the dataset will be partitioned into:
- **100 farmer nodes** (simulating 100 farmers/agents)
- **Non-IID distribution** (each farmer has different crop types/regions)
- **Varying data sizes** (10-150 images per farmer, mimicking real-world conditions)

See `src/simulation/data_partitioner.py` for implementation details.

## Alternative Datasets

### 1. PlantVillage Dataset
- **Source**: https://www.kaggle.com/datasets/emmarex/plantdisease
- **Use case**: Crop disease classification
- **Size**: 54,000+ images, 38 classes

### 2. Global Wheat Head Detection
- **Source**: https://www.kaggle.com/c/global-wheat-detection
- **Use case**: Wheat head detection
- **Size**: 3,400+ images

### 3. Indonesia-Specific Options
Consider creating a custom dataset by scraping:
- Indonesian agricultural research institutions
- Local farmer cooperatives
- Government agricultural databases (Kementerian Pertanian)

## Troubleshooting

### Issue: Kaggle API not working
**Solution**: 
1. Generate Kaggle API token from kaggle.com/account
2. Place in `~/.kaggle/kaggle.json`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Issue: Dataset too large for storage
**Solution**: 
1. Use sample subset (10% of data) for initial experiments
2. Use `--sample-size 1000` flag in download script

### Issue: Wrong annotation format
**Solution**: 
Use the format converter:
```bash
python src/simulation/convert_annotations.py --format coco --output yolo
```