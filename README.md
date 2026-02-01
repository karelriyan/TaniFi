# TaniFi: Federated Learning for Resource-Constrained Agricultural Networks

## Research Paper Implementation
**Title:** Simulation of Bandwidth-Efficient Federated Learning Architectures for Resource-Constrained Agricultural Networks in Indonesia

## Project Overview
TaniFi is a decentralized AI system that enables farmers in Indonesia to collaboratively train crop disease detection models using DiLoCo (Distributed Low-Communication) federated learning, deployed on Base L2 blockchain.

### Key Features
- **Bandwidth-Efficient Training**: DiLoCo algorithm with local training (500 steps) before synchronization
- **Edge Computing**: LoRA adapters for low-resource devices (2GB RAM Android phones)
- **Blockchain Incentives**: Smart contracts on Base L2 for proof-of-learning and token distribution
- **YOLOv11 Foundation**: 99.8% accuracy crop disease detection

## Project Structure
```
tanifi-federated-learning/
├── data/
│   ├── raw/              # Raw datasets (WeedsGalore, Global Wheat)
│   └── processed/        # Preprocessed data for training
├── models/
│   └── checkpoints/      # Model weights and LoRA adapters
├── src/
│   ├── simulation/       # DiLoCo simulation scripts
│   └── contracts/        # Smart contracts (Solidity)
├── experiments/
│   └── results/          # Experiment outputs and metrics
├── notebooks/            # Jupyter notebooks for analysis
└── docs/                 # Paper drafts and documentation
```

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Download
See `docs/dataset_setup.md` for WeedsGalore dataset instructions

### 3. Run Simulation
```bash
python src/simulation/diloco_trainer.py --config experiments/config.yaml
```

## Research Milestones

### Week 1 (Current)
- [x] Repository setup
- [ ] WeedsGalore dataset download
- [ ] Basic DiLoCo simulation script

### Week 2
- [ ] Base L2 smart contract design
- [ ] Proof-of-learning mechanism

### Week 3+
- [ ] Full federated learning experiments
- [ ] Economic simulation (tokenomics)
- [ ] Paper drafting

## Citation
```bibtex
@article{tanifi2025,
  title={Simulation of Bandwidth-Efficient Federated Learning Architectures for Resource-Constrained Agricultural Networks in Indonesia},
  author={[Your Name]},
  journal={[Target Journal]},
  year={2025}
}
```

## License
MIT License - See LICENSE file for details