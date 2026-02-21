# Benchmarking script for LoRA vs QLoRA adapters
"""
Runs a short training session for both LoRA and QLoRA adapters on the same dataset
and records key metrics (training time, final validation accuracy, bandwidth saved).
The script is intended for quick performance comparison and can be extended for
more thorough benchmarking.
"""

import time
import json
from pathlib import Path
# Import main_training with fallback for script execution
try:
    from .diloco_trainer import main_training
except ImportError:  # pragma: no cover
    from src.simulation.diloco_trainer import main_training

def run_benchmark(adapter_type: str, config_overrides: dict = None):
    """Run a single training run with the specified adapter type.

    Parameters
    ----------
    adapter_type: str
        Either "lora" or "qlora".
    config_overrides: dict, optional
        Additional configuration values to merge into the default config.
    """
    # Load default config from main_training (it will use the built‑in defaults)
    config = {
        "num_farmers": 5,
        "local_steps": 50,
        "total_rounds": 5,
        "warmup_rounds": 1,
        "img_size": 64,
        "batch_size": 16,
        "num_epochs_baseline": 2,
        "adapter_type": adapter_type,
    }
    if config_overrides:
        config.update(config_overrides)

    start = time.time()
    # Run federated training (centralized=False) with the given adapter
    metrics = main_training(
        config_file=None,
        centralized=False,
        real_data=False,  # use synthetic data for speed
        save_plots=False,
    )
    duration = time.time() - start
    return {
        "adapter_type": adapter_type,
        "duration_sec": duration,
        "global_metrics": metrics,
    }

def main():
    results = []
    for adapter in ["lora", "qlora"]:
        print(f"\nRunning benchmark for {adapter.upper()} adapter...")
        res = run_benchmark(adapter)
        results.append(res)

    # Save results to a JSON file for later analysis
    results_dir = Path(__file__).parent.parent.parent / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"benchmark_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to {out_path}")

if __name__ == "__main__":
    main()
