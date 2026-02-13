# TaniFi: Federated Learning for Resource-Constrained Agricultural Networks

## Digital Farming Revolution on the Edge

TaniFi is a research project simulating federated learning architectures for bandwidth-constrained agricultural networks, specifically designed for Indonesian agricultural 4.0 environments. Built for the research paper on decentralized AI for smart farming.

## ⚡ Quick Start

```bash
# 1. Verify environment setup (checks dependencies, dataset, model)
python3 verify_setup.py

# 2. Check data structure
ls -la data/weedsgalore/  # Should contain weedsgalore-dataset folder

# 3. Run experiments
python3 src/simulation/diloco_trainer.py --config experiments/config.yaml --real-data

# 4. Run centralized baseline for comparison
python3 src/simulation/diloco_trainer.py --config experiments/config.yaml --real-data --centralized
```

## 📊 Research Context

This project evaluates DiLoCo (Distributed Low-Communication) protocol in the context of:
- **Limited bandwidth**: Farm areas with unstable 3G/4G connectivity
- **Edge resources**: Devices with modest computational power
- **Privacy preserving**: Data remains on farmer devices
- **Data heterogeneity**: 100+ distributed farm nodes

### Key Features
- ✅ LoRA adapter shards for efficient communication
- ✅ Non-IID data distribution across farmers
- ✅ Real agricultural dataset (WeedsGalore) OR synthetic data for testing
- ✅ Centralized baseline comparison
- ✅ Comprehensive metrics: Accuracy, F1-Macro, bandwidth savings
- ✅ Local training with configurable rounds and steps

## 📁 Project Architecture

```
TaniFi/
├── src/simulation/
│   ├── diloco_trainer.py          # Main federated learning coordinator
│   └── weedsgalore_loader.py      # Dataset loader (real labels from masks)
│
├── data/
│   ├── README.md                  # This file - data structure guide
│   └── weedsgalore/               # WeedsGalore dataset
│       └── weedsgalore-dataset/
│           ├── 2023-05-25/        # Date-based folder structure
│           ├── 2023-05-30/
│           ├── splits/            # train.txt / val.txt / test.txt
│           └── ... (more date folders)
│
├── models/
│   └── checkpoints/               # Trained model weights
│
├── experiments/
│   ├── config.yaml                # Default experiment configuration
│   ├── config_10f_20r_500s.yaml   # 10 farmers, 20 rounds, 500 steps
│   ├── config_10f_200r_50s.yaml   # 10 farmers, 200 rounds, 50 steps
│   ├── run_experiments.py         # Automated experiment runner
│   ├── generate_configs.py        # Config generator for parameter sweeps
│   └── results/                   # JSON metrics and plots
│       ├── plots/                 # PNG graphs
│       └── tables/                # CSV results
│
├── notebooks/
│   └── analysis_template.ipynb    # Jupyter analysis notebook
│
├── yolo11n-cls.pt                 # YOLOv11 model (required)
├── verify_setup.py                # Environment verification script
├── requirements.txt               # Python dependencies
├── LICENSE
└── README.md                      # This file
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- GPU recommended for training (CPU works but slower)
- 4GB+ RAM, 5GB+ disk space

### Installation
```bash
# Clone repository
git clone <repo-url> && cd TaniFi

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# YOLO model (automatic download by ultralytics)
# The model downloads on first run
```

## ⚠️ Dataset Setup - WeedsGalore

**Important**: For real experiments, you need the WeedsGalore dataset.

```bash
# Option 1: Quick test with synthetic data (no download needed)
python3 src/simulation/diloco_trainer.py --config experiments/config.yaml

# Option 2: Real data from Kaggle
# 1. Visit: https://www.kaggle.com/datasets/vinayakshanawad/weedsgalore
# 2. Download and extract to: data/weedsgalore/weedsgalore-dataset/
#
# Structure:
# data/weedsgalore/weedsgalore-dataset/
# ├── 2023-05-25/ (images, semantics folders)
# ├── splits/ (train.txt, val.txt, test.txt)
# └── ... more date folders
```

## 🚀 Usage

### 1. DiLoCo Federated Learning
```bash
python3 src/simulation/diloco_trainer.py \
    --real-data \
    --config experiments/config_10f_20r_500s.yaml
```

### 2. Centralized Baseline
```bash
python3 src/simulation/diloco_trainer.py \
    --real-data \
    --centralized \
    --config experiments/config.yaml
```

### 3. Custom Config
```bash
python3 src/simulation/diloco_trainer.py \
    --real-data \
    --num-farmers 5 \
    --total-rounds 30 \
    --local-steps 200
```

## 📦 Structure Cleaned

✅ **Completed Refactoring:**
- Removed `data/processed/` (unused)
- Renamed `data/raw/` → `data/weedsgalore/`
- Removed `src/contracts/` (empty)
- Removed `models/checkpoints/` instructions (was empty)
- Fixed all path references in code
- Cleaned up README.md

🔍 **Verify:**
```bash
ls -la data/weedsgalore/weedsgalore-dataset/
# Should show date folders and splits/
```

## 🚀 Ready to Experiment

After verifying `weedsgalore-dataset/` exists:

```bash
# Single experiment
python3 src/simulation/diloco_trainer.py --real-data

# Compare with centralized
python3 src/simulation/diloco_trainer.py --real-data --centralized
```

**Results saved to**: `experiments/results/`

---

**Status**: ✅ Refactored & Ready