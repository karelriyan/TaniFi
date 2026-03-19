# TaniFi: Federated Learning for Resource-Constrained Agricultural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

TaniFi (short for **Tani Federated Intelligence**) is a research-focused open-source simulation framework for evaluating federated learning architectures in bandwidth-constrained environments, specifically designed for "Agriculture 4.0" in developing regions. 

It evaluates methods like **DiLoCo** (Distributed Low-Communication) and **QLoRA** adapter tuning against traditional baselines (Centralized training and FedAvg) for vision-based plant disease classification tasks.

---

## 📖 Table of Contents
- [Project Goals](#-project-goals)
- [Key Features](#-key-features)
- [Architecture Overview](#-architecture-overview)
- [Getting Started](#-getting-started)
- [Using the Datasets](#-using-the-datasets)
- [Configuration and Experiments](#-configuration-and-experiments)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Goals

The main objective of TaniFi is to provide a robust, modular, and extensible platform for:
1. Simulating distributed agricultural sensor networks ("Farmer Nodes").
2. Evaluating communication-efficient federated learning strategies (bandwidth savings > 98%).
3. Comparing model performance (F1-Macro, Accuracy) across varied local training step counts and synchronization frequencies.

---

## ✨ Key Features

- **Resource-Efficient Strategy**: Native integration of parameter-efficient fine-tuning (PEFT) methods, specifically **LoRA** and integrated **4-bit QLoRA**.
- **DiLoCo Simulation**: Implements robust outer-step optimization (Nesterov momentum) at the central coordinator level to reduce required communication rounds dramatically over standard FedAvg.
- **Agricultural Datasets**: Turn-key support for processing the `WeedsGalore` dataset (with semantic mask translation) and the `PlantVillage` dataset.
- **Experiment Tracking**: Automatic JSON metric logging, comparative VRAM footprint analysis, and convergence plot generation.

---

## 🏗️ Architecture Overview

TaniFi decouples simulation logic into clean, distinct modules:

*   **`coordinator.py`**: The central server node. Aggregates adapter weights (shards) broadcast by farmers using Pseudo-Gradient methods, applies server-side momentum, and tracks global metrics.
*   **`farmer.py`**: The remote edge nodes. Handles instantiation of the PEFT adapters on base models, dynamic local training epochs, and gradient clipping constraints.
*   **`model.py` / `adapters.py`**: Wrappers for the YOLOv11-Nano base backbone and PEFT configuration generation.
*   **`data.py`**: Factory-pattern loaders that handle image augmentations and quality filters (blur variance/green ratio checks).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher.
- A CUDA-compatible GPU is strongly recommended.
- Git.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/TaniFi.git
cd TaniFi

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# 3. Install required packages
pip install -r requirements.txt

# 4. Verify the setup
python src/simulation/verify_setup.py
```

### Running a Minimal Working Example

You can run a fast, dummy-data simulation out-of-the-box to verify that the training loop and tensor operations are working correctly:

```bash
# Run a brief simulation using synthetic data and the DiLoCo protocol
python src/simulation/diloco_trainer.py --config experiments/quick_test_config.yaml
```

Check the `experiments/results/` directory for the output logs and plots.

---

## 💾 Using the Datasets

> **Note:** Due to size constraints, datasets **must be downloaded externally**. 

We currently support the `WeedsGalore` and `PlantVillage` datasets.

1.  Download the datasets into the `data/` directory.
2.  Do **not** commit these files; they are included in the `.gitignore`.

For detailed instructions on dataset directory structure and downloading, see the **[Data Documentation](data/README.md)**.

---

## 🧪 Configuration and Experiments

TaniFi uses declarative YAML configuration files to define experiment parameters (number of farmers, local steps, adapter settings).

Example of running a full experiment with real data:
```bash
python experiments/run_experiments.py --config experiments/config_diloco_qlora.yaml --real-data
```

For a comprehensive guide on creating experiment YAMLs, analyzing the JSON output results, and generating comparison plots, please see the **[Experiments Documentation](experiments/README.md)**.

---

## 🗺️ Roadmap

- [x] Initial Simulation Loop implementation.
- [x] DiLoCo momentum outer-optimization.
- [x] QLoRA 4-bit integration.
- [ ] Asynchronous farmer capability (simulating stragglers/offline nodes).
- [ ] Edge device resource tracking (estimated milliJoules/battery drain).
- [ ] Integration with non-vision-based sensor datasets.

---

## 🤝 Contributing

We welcome contributions to TaniFi! Whether it's adding a new dataset loader, a novel aggregation protocol, or improving documentation.

Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started.

---

## 📜 License

TaniFi is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

*Note: The associated research paper documenting the core methodology and real-world applicability findings in Indonesia can be found in `docs/Paper/mainTemplate_JIKI.pdf`.*