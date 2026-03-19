# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-03-XX

### Added
- **Federated Learning Simulation**: Full implementation of DiLoCo (Distributed Low-Communication) for resource-constrained environments.
- **Models**: Integration of YOLOv11-Nano with custom classification heads.
- **Adapters**: Support for LoRA and QLoRA efficient fine-tuning techniques via PEFT.
- **Data Loaders**: Native support for WeedsGalore plant datasets and auto-label parsing.
- **Experiment Tracking**: JSON logging, metric charting, and LaTeX table generation for paper-ready results.
- **CI/CD Boilerplate**: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, and comprehensive `README.md`.
- **Unit Testing**: Initial test suite for data loaders to ensure reproducibility.

### Changed
- Project structure completely refactored from initial script iterations into a modular `src/simulation` package (`coordinator.py`, `farmer.py`, etc.).
- Relocated large `.pt` checkpoint files, unneeded bytecode caches, and inline datasets.
- Migrated hardcoded system paths to relative paths utilizing `pathlib` for broad machine compatibility.

### Fixed
- Early bugs with unconstrained VRAM utilization during QLoRA 4-bit quantization layers.
- Metric aggregation in Centralized baseline mode.
