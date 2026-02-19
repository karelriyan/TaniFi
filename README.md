# TaniFi: Federated Learning for Resource-Constrained Agricultural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌱 Digital Farming Revolution on the Edge

TaniFi (Tani Federated Intelligence) is a research project simulating federated learning architectures for bandwidth-constrained agricultural networks, specifically designed for Indonesian agricultural 4.0 environments. This project implements the DiLoCo (Distributed Low-Communication) protocol to enable efficient AI model training across distributed farm nodes with limited connectivity.

## 📚 Table of Contents
- [Key Features](#-key-features)
- [Installation & Setup](#-installation--setup)
- [Dataset Configuration](#%EF%B8%8F-dataset-setup)
- [Running Experiments](#-running-experiments)
- [Project Architecture](#-project-architecture)
- [Results & Evaluation](#-results--evaluation)
- [Contributing](#-contributing)
- [License](#-license)

## ✅ Key Features

- **Bandwidth-Efficient Communication**
  - LoRA adapter shards for efficient parameter updates
  - 98%+ bandwidth reduction compared to full model transfer

- **Real Agricultural Dataset**
  - Supports WeedsGalore dataset with automatic label derivation
  - Synthetic data option for quick testing

- **Flexible Experimentation**
  - Configurable farmers, rounds, and local steps
  - Centralized baseline for performance comparison
  - Comprehensive metrics: Accuracy, F1-Macro, bandwidth savings

- **Reproducible Research**
  - Automated experiment runner
  - Jupyter analysis templates
  - LaTeX paper templates

## 🆕 New Features

- **QLoRA Adapter Integration**
  - Added `QLoRAAdapter` wrapper for PEFT‑generated quantized models.
  - Automatic 4‑bit quantization of linear layers using `bitsandbytes`.
  - Unified adapter interface (`get_adapter_params`, `set_adapter_params`).
  - CLI flag `--adapter-type` to select between `lora` and `qlora`.
  - Optional JSON config via `--adapter-config` for custom adapter settings.

- **Benchmarking Script**
  - New `src/simulation/benchmark.py` runs short training sessions for both adapters.
  - Records execution time, global metrics, and saves results to `experiments/results/`.
  - Provides a quick way to compare performance and bandwidth savings.

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- GPU recommended for training (CPU supported but slower)
- 4GB+ RAM, 5GB+ disk space

### Step-by-Step Installation
```bash
# 1. Clone repository
git clone https://github.com/your-username/TaniFi.git
cd TaniFi

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python verify_setup.py
```

The YOLOv11 model (`yolo11s-cls.pt`) will be automatically downloaded on first run.

## ⚠️ Dataset Setup

### WeedsGalore Dataset
For real experiments, download the WeedsGalore dataset:

```bash
# Manual download (recommended):
# 1. Visit: https://www.kaggle.com/datasets/vinayakshanawad/weedsgalore
# 2. Download and extract to: data/weedsgalore/weedsgalore-dataset/

# Automated download (requires Kaggle API):
python src/simulation/download_dataset.py --dataset weedsgalore
```

### Required Dataset Structure
```
data/weedsgalore/weedsgalore-dataset/
├── 2023-05-25/
│   ├── images/       # Contains _R.png, _G.png, _B.png files
│   └── semantics/    # Contains semantic masks (.png)
├── 2023-05-30/
├── ...
└── splits/           # Contains train.txt, val.txt, test.txt
```

### Verification
```bash
ls -la data/weedsgalore/weedsgalore-dataset/
# Should show date folders and splits/
```

### PlantVillage Dataset
The project also supports the PlantVillage dataset for disease classification.

**Structure:**
```
data/archive/PlantVillage_for_object_detection/Dataset/
├── images/       # All images
├── labels/       # YOLO format labels (.txt)
└── classes.yaml  # Class names
```

**Configuration:**
To use PlantVillage, update your config file:
```yaml
dataset:
  name: plantvillage
  image_size: 224 # Recommended size
```

## 🚀 Running Experiments

### 1. Federated Learning (DiLoCo)
```bash
python src/simulation/diloco_trainer.py \
    --real-data \
    --config experiments/config_10f_20r_500s.yaml
```

### 2. Centralized Baseline
```bash
python src/simulation/diloco_trainer.py \
    --real-data \
    --centralized \
    --config experiments/config.yaml
```

### 3. Custom Configuration
```bash
python src/simulation/diloco_trainer.py \
    --real-data \
    --num-farmers 5 \
    --total-rounds 30 \
    --local-steps 200
```

### 4. Parameter Sweep
```bash
# Generate multiple configurations
python experiments/generate_configs.py

# Run all experiments
python experiments/run_experiments.py
```

## 📁 Project Architecture

```
TaniFi/
├── data/                       # Agricultural datasets
│   └── weedsgalore/            # WeedsGalore dataset
│       └── weedsgalore-dataset/
│           ├── 2023-05-25/     # Date-based folders
│           ├── splits/         # Dataset splits
│           └── ...
│
├── docs/                       # Research documentation
│   └── Paper/                  # LaTeX paper templates
│       ├── AuthorGuideline_JIKI/
│       └── Paper-JIKI-Latex-Karel/
│
├── experiments/                # Experiment management
│   ├── configs/                # Configuration files
│   ├── results/                # Experiment outputs
│   ├── generate_configs.py     # Config generator
│   └── run_experiments.py      # Experiment runner
│
├── notebooks/                  # Data analysis
│   └── analysis_template.ipynb # Jupyter analysis
│
├── src/simulation/             # Core simulation code
│   ├── diloco_trainer.py       # Main training coordinator
│   ├── weedsgalore_loader.py   # WeedsGalore loader
│   ├── plantvillage_loader.py  # PlantVillage loader
│   └── image_filters.py        # Quality control filters
│
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── verify_setup.py             # Environment verification
```

## 📊 Results & Evaluation

Experiment results are saved in `experiments/results/` with:
- JSON files containing detailed metrics
- PNG plots of training progress
- LaTeX tables for paper inclusion

To analyze results:
```bash
jupyter notebook notebooks/analysis_template.ipynb
```

Key metrics include:
- **Communication Efficiency**: Bandwidth savings percentage
- **Model Performance**: Accuracy, F1-Macro, loss curves
- **Scalability**: Performance vs. number of farmers

## 👥 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Research Paper**: [Simulation of Bandwidth-Efficient Federated Learning Architectures for Resource-Constrained Agricultural Networks in Indonesia](docs/Paper/Paper-JIKI-Latex-Karel/mainTemplate_JIKI.pdf)

**Status**: ✅ Production Ready | 🔬 Research Grade