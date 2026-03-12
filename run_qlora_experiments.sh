#!/bin/bash
# ============================================================
# TaniFi — Run 3 QLoRA Scenarios
# Hardware: RTX 5050 8GB VRAM, 20 CPU cores, 16GB RAM
# Optimized: batch_size=64, num_workers=8, FAST_MODE=1
# ============================================================

set -e  # Stop on first error

PYTHON="./venv/bin/python"
EXPERIMENTS_DIR="experiments"

echo "============================================================"
echo " TaniFi QLoRA Experiments"
echo " Scenarios: Centralized | FedAvg | DiLoCo"
echo "============================================================"
echo ""

# ── Scenario 1: Centralized + QLoRA ─────────────────────────
echo "▶ [1/3] Centralized + QLoRA"
echo "   (baseline — trains a single model with QLoRA on all data)"
echo ""

FAST_MODE=1 $PYTHON src/simulation/diloco_trainer.py \
  --config $EXPERIMENTS_DIR/config_centralized_qlora.yaml \
  --centralized \
  --adapter-type qlora \
  --real-data

echo ""
echo "✅ Scenario 1 done."
echo ""

# ── Scenario 2: FedAvg + QLoRA ──────────────────────────────
echo "▶ [2/3] FedAvg + QLoRA"
echo "   (10 farmers, 200 rounds, 50 local steps — frequent sync)"
echo ""

FAST_MODE=1 $PYTHON src/simulation/diloco_trainer.py \
  --config $EXPERIMENTS_DIR/config_fedavg_qlora.yaml \
  --adapter-type qlora \
  --real-data

echo ""
echo "✅ Scenario 2 done."
echo ""

# ── Scenario 3: DiLoCo + QLoRA ──────────────────────────────
echo "▶ [3/3] DiLoCo + QLoRA"
echo "   (10 farmers, 20 rounds, 500 local steps — rare sync)"
echo ""

FAST_MODE=1 $PYTHON src/simulation/diloco_trainer.py \
  --config $EXPERIMENTS_DIR/config_diloco_qlora.yaml \
  --adapter-type qlora \
  --real-data

echo ""
echo "✅ Scenario 3 done."
echo ""
echo "============================================================"
echo " All 3 scenarios complete! Results saved to experiments/results/"
echo "============================================================"
