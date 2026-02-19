# Simulation Package Overview

This document provides a concise description of each module in the `src/simulation` package after the recent refactor.

| File | Description |
|------|-------------|
| `__init__.py` | Marks the directory as a Python package, enabling relative imports across the simulation modules. |
| `model.py` | Defines the `YOLOv11ClassificationModel` class, a wrapper around the YOLOv11‑Nano backbone with a custom classification head. This model is used for both centralized and federated training. |
| `data.py` | Supplies dataset creation utilities:
- `create_weedsgalore_dataset` builds a real dataset from the WeedsGalore images with appropriate augmentations.
- `create_plantvillage_dataset` builds the PlantVillage dataset with standard preprocessing.
- `create_dataset` acts as a unified factory to create either WeedsGalore, PlantVillage, or synthetic datasets based on configuration. |
| `weedsgalore_loader.py` | Handles the date-based folder structure and semantic mask processing for the WeedsGalore dataset. Includes logic for channel merging and label derivation. |
| `plantvillage_loader.py` | Handles the PlantVillage dataset structure. Supports YOLO-format labels and standard train/val/test splits. |
| `image_filters.py` | Implementation of quality control checks for images. Detects blur (Laplacian variance) and verifies plant content (green pixel ratio) to filter out poor quality data during loading. |
| `evaluation.py` | Implements evaluation helpers:
- `evaluate_model` runs inference on a dataloader and returns loss, accuracy, and macro‑F1.
- `compute_class_weights` calculates inverse‑frequency class weights for weighted loss functions. |
| `utils.py` | Contains shared constants and small utilities:
- `FAST_MODE` toggles a lightweight training configuration.
- `_make_progress_bar` builds a simple textual progress bar used by the coordinator and trainer. |
| `farmer.py` | Implements the `FarmerNode` class, representing a single participant in the federated simulation. Handles local training, optimizer setup, AMP support, and shard extraction/updating. |
| `coordinator.py` | Implements the `DiLoCoCoordinator` class, which manages the overall federated learning loop: initializing farmers, aggregating shards with weighted averaging, applying Nesterov momentum updates, checkpointing, and final evaluation. |
| `training.py` | Central orchestration module:
- `train_centralized_baseline` runs a standard (non‑federated) training loop for baseline comparison.
- `main_training` parses configuration, creates datasets, instantiates the model, and either runs the baseline or launches the federated `DiLoCoCoordinator`. Results are saved and optional plots generated. |
| `diloco_trainer.py` | Thin entry‑point script exposing a CLI. It parses arguments (`--config`, `--centralized`, `--real-data`, `--no-plots`) and forwards execution to `training.main_training`. |
| `benchmark.py` | Small utility to benchmark LoRA vs QLoRA adapters by invoking `main_training` with synthetic data and recording runtime and metrics. |
| `adapters.py` | (Unchanged) Provides the `AdapterFactory` and adapter implementations (LoRA, QLoRA) used by `FarmerNode`. |

The refactor separates concerns, making the codebase easier to understand, test, and extend. Each component now lives in its own module with clear responsibilities and minimal cross‑module coupling.
