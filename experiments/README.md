# Experiment Management

The `experiments/` directory is the core control center for running federated learning simulations in TaniFi. It includes predefined configuration configurations, scripts to batch-process simulations, and a historical store of experiment results.

## Configuration Files (`*.yaml`)

Experiments are controlled declaratively using YAML files. This ensures reproducibility and separation of code from parameters.

### Example Configuration:
```yaml
experiment_name: "DiLoCo_QLoRA_10Farmers_500Steps"
dataset:
  name: "weedsgalore" # or 'plantvillage' or 'dummy'
  image_size: 224

federated:
  enabled: true       # False will trigger the Centralized Baseline mode
  protocol: "diloco"  # 'diloco' or 'fedavg'
  num_farmers: 10
  total_rounds: 30
  local_steps: 500

adapter:
  type: "qlora"       # 'lora' or 'qlora'
  r: 16
  alpha: 32
  dropout: 0.1
```

### Included Configs
- **`config_centralized_qlora.yaml`**: Standard non-federated baseline collecting all edge data to a single node.
- **`config_fedavg_qlora.yaml`**: The standard Federated Averaging algorithm baseline.
- **`config_diloco_qlora.yaml`**: Our primary novel architecture, Distributed Low-Communication with 4-bit quantization.
- **`quick_test_config.yaml`**: A fast-running, low-parameter config using synthetic data to verify system integrity before launching multi-hour runs.

## Running Experiments

You can execute a single experiment manually via the core trainer:
```bash
python src/simulation/diloco_trainer.py --config experiments/config_diloco_qlora.yaml --real-data
```

Or you can batch-process multiple configurations sequentially using our runner snippet:
```bash
# This script will automatically iterate through defined YAML files in the directory
python experiments/run_experiments.py
```

## Results Directory

After an experiment completes, the data is saved automatically in `experiments/results/`.
By default this will generate:
1. **JSON Logs**: e.g., `federated_diLoCo_20260221_222214.json`. Contains global metrics per round (Accuracy, F1, Loss, Server Communication Megabytes).
2. **Metadata**: Hard-recorded VRAM utilizations.
3. **Plots (`plots/`)**: Auto-generated visualizations comparing convergence speeds.

### Archiving Runs
Historical or partial runs are periodically backed up into the `results_history/` sub-folder to keep the main output directory clean.
