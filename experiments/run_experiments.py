#!/usr/bin/env python3
"""
TaniFi Parallel Experiment Runner

This script runs multiple DiLoCo federated learning experiments in parallel
or sequential mode. It provides clear progress tracking and execution logs.

Usage:
    python run_experiments.py                    # Sequential (1 worker)
    python run_experiments.py --workers 4        # Parallel with 4 workers
    python run_experiments.py --sequential       # Force sequential mode
    python run_experiments.py --real-data        # Use WeedsGalore dataset
    python run_experiments.py --dummy-data       # Use dummy dataset (default)

Output:
    - Real-time progress with clear experiment identification
    - execution_log.json with detailed results
"""

import subprocess
import time
import json
import signal
import sys
import re
from pathlib import Path
from datetime import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# This section controls which dataset is used for ALL experiments.
#
# HOW TO SWITCH DATASETS:
# -----------------------
# Option 1: Change the DATASET_TYPE variable below
# Option 2: Use command line flag: --real-data or --dummy-data
#
# AVAILABLE DATASET TYPES:
# ------------------------
# "dummy"       - Random generated data (fast, for testing pipeline)
# "weedsgalore" - WeedsGalore agricultural dataset (real research data)
# "custom"      - Custom dataset (requires implementing your own loader)
#
# TO ADD A NEW DATASET:
# ---------------------
# 1. Create a loader in src/simulation/ (e.g., my_dataset_loader.py)
#    - Implement a Dataset class with __len__ and __getitem__ methods
#    - __getitem__ should return (image_tensor, label) tuple
#
# 2. Add the dataset to src/simulation/diloco_trainer.py:
#    - Import your loader in create_dataset() function
#    - Add a new condition for your dataset type
#
# 3. Add the dataset type name to SUPPORTED_DATASETS below
#
# 4. Run experiments with: python run_experiments.py --dataset your_dataset_name
#
# REPRODUCIBILITY:
# ----------------
# - Random seed is set in diloco_trainer.py (default: 42)
# - Same dataset type + same seed = reproducible results
# - Document which dataset you used in your research paper
#
# =============================================================================

# Supported dataset types (add new datasets here)
SUPPORTED_DATASETS = ["dummy", "weedsgalore"]

# Default dataset type - CHANGE THIS TO SWITCH DATASETS FOR ALL EXPERIMENTS
DATASET_TYPE = "weedsgalore"  # <-- CHANGE THIS: "dummy" or "weedsgalore"


# =============================================================================
# END OF CONFIGURATION
# =============================================================================

# Global lock for thread-safe printing
print_lock = Lock()

# Global counter for completed experiments
completed_count = 0
total_count = 0


def extract_config_info(config_path):
    """Extract farmers, rounds, steps from config filename."""
    filename = Path(config_path).stem
    # Pattern: config_50f_15r_500s or similar
    match = re.match(r'config_(\d+)f_(\d+)r_(\d+)s', filename)
    if match:
        return {
            'farmers': int(match.group(1)),
            'rounds': int(match.group(2)),
            'steps': int(match.group(3))
        }
    return None


def format_config_info(config_path):
    """Format config info for display."""
    info = extract_config_info(config_path)
    if info:
        return f"{info['farmers']}f/{info['rounds']}r/{info['steps']}s"
    return Path(config_path).stem


def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()


def run_single_experiment(config_path, experiment_index, total_experiments,
                          timeout_seconds=86400, dataset_type="dummy"):
    """
    Run a single experiment and track execution time.

    Args:
        config_path: Path to the config file
        experiment_index: Index of this experiment (for progress tracking)
        total_experiments: Total number of experiments
        timeout_seconds: Maximum time allowed per experiment
        dataset_type: Dataset to use ("dummy" or "weedsgalore")

    Returns:
        Dict with execution results (success, duration, etc.)
    """
    global completed_count

    config_file = Path(config_path)
    config_info = format_config_info(config_path)

    if not config_file.exists():
        return {
            'config': str(config_path),
            'config_info': config_info,
            'dataset': dataset_type,
            'success': False,
            'error': 'Config file not found',
            'duration': 0
        }

    safe_print(f"\n{'='*70}")
    safe_print(f"▶ STARTING [{experiment_index}/{total_experiments}]: {config_file.name}")
    safe_print(f"  Config: {config_info}")
    safe_print(f"  Dataset: {dataset_type}")
    safe_print(f"{'='*70}")

    start_time = time.time()

    try:
        # Get absolute paths
        config_absolute = config_file.absolute()

        # Run from src/simulation directory
        simulation_dir = Path(__file__).parent.parent / 'src' / 'simulation'

        # Build command with dataset flag
        cmd = [
            'python3',
            'diloco_trainer.py',
            '--config',
            str(config_absolute)
        ]

        # Add dataset flag based on type
        if dataset_type == "weedsgalore":
            cmd.append('--real-data')
        else:
            cmd.append('--dummy-data')

        # Run diloco_trainer with specified config and dataset
        # Stream output in real-time so user can see progress bars
        process = subprocess.Popen(
            cmd,
            cwd=str(simulation_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Collect output while streaming
        output_lines = []
        try:
            while True:
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    process.kill()
                    raise subprocess.TimeoutExpired(cmd, timeout_seconds)

                line = process.stdout.readline()
                if line:
                    output_lines.append(line)
                    # Print progress lines (contains progress bar or key info)
                    if any(x in line for x in ['PROGRESS:', 'Training farmers:', '✅', '❌', '🔄', 'Round', 'ETA:']):
                        print(line, end='', flush=True)
                elif process.poll() is not None:
                    break

            # Get remaining output
            remaining = process.stdout.read()
            if remaining:
                output_lines.append(remaining)

        except subprocess.TimeoutExpired:
            raise

        output_text = ''.join(output_lines)
        returncode = process.returncode

        end_time = time.time()
        duration = end_time - start_time

        success = returncode == 0
        completed_count += 1

        if success:
            safe_print(f"\n{'='*70}")
            safe_print(f"✅ COMPLETED [{completed_count}/{total_experiments}]: {config_file.name}")
            safe_print(f"   Config: {config_info}")
            safe_print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
            safe_print(f"   Progress: {completed_count}/{total_experiments} ({100*completed_count/total_experiments:.0f}%)")
            safe_print(f"{'='*70}")
        else:
            safe_print(f"\n{'='*70}")
            safe_print(f"❌ FAILED [{experiment_index}/{total_experiments}]: {config_file.name}")
            safe_print(f"   Config: {config_info}")
            safe_print(f"   Return code: {returncode}")
            # Show last 300 chars of output for error info
            error_snippet = output_text[-500:] if len(output_text) > 500 else output_text
            safe_print(f"   Error: {error_snippet[:300]}...")
            safe_print(f"{'='*70}")

        return {
            'config': config_file.name,
            'config_info': config_info,
            'dataset': dataset_type,
            'success': success,
            'duration': duration,
            'duration_minutes': duration / 60,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'stdout': output_text if success else None,
            'stderr': output_text if not success else None
        }

    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time

        safe_print(f"\n{'='*70}")
        safe_print(f"⏱️ TIMEOUT [{experiment_index}/{total_experiments}]: {config_file.name}")
        safe_print(f"   Config: {config_info}")
        safe_print(f"   Dataset: {dataset_type}")
        safe_print(f"   Duration: {duration:.1f}s (exceeded {timeout_seconds}s limit)")
        safe_print(f"{'='*70}")

        return {
            'config': config_file.name,
            'config_info': config_info,
            'dataset': dataset_type,
            'success': False,
            'error': f'Timeout after {timeout_seconds}s',
            'duration': duration,
            'duration_minutes': duration / 60,
            'skipped': True
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        safe_print(f"\n{'='*70}")
        safe_print(f"❌ EXCEPTION [{experiment_index}/{total_experiments}]: {config_file.name}")
        safe_print(f"   Config: {config_info}")
        safe_print(f"   Dataset: {dataset_type}")
        safe_print(f"   Error: {str(e)}")
        safe_print(f"{'='*70}")

        return {
            'config': config_file.name,
            'config_info': config_info,
            'dataset': dataset_type,
            'success': False,
            'error': str(e),
            'duration': duration,
            'duration_minutes': duration / 60
        }


def run_experiments_parallel(config_files, num_workers=2, timeout_seconds=86400,
                             dataset_type="dummy"):
    """
    Run multiple experiments in parallel.

    Args:
        config_files: List of config file paths
        num_workers: Number of parallel workers
        timeout_seconds: Timeout per experiment
        dataset_type: Dataset to use for all experiments
    """
    global completed_count, total_count
    completed_count = 0
    total_count = len(config_files)

    safe_print(f"\n{'#'*70}")
    safe_print(f"# PARALLEL EXECUTION")
    safe_print(f"#   Total experiments: {len(config_files)}")
    safe_print(f"#   Parallel workers: {num_workers}")
    safe_print(f"#   Dataset: {dataset_type}")
    safe_print(f"#   Timeout per experiment: {timeout_seconds}s ({timeout_seconds/60:.0f} min)")
    safe_print(f"{'#'*70}\n")

    # List all experiments
    safe_print("Experiments to run:")
    for idx, config in enumerate(config_files, 1):
        info = format_config_info(config)
        safe_print(f"  {idx:2d}. {Path(config).name} ({info})")
    safe_print()

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_config = {
            executor.submit(
                run_single_experiment, config, idx + 1, len(config_files),
                timeout_seconds, dataset_type
            ): config
            for idx, config in enumerate(config_files)
        }

        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                safe_print(f"❌ Exception for {config}: {str(e)}")
                results.append({
                    'config': str(config),
                    'config_info': format_config_info(config),
                    'dataset': dataset_type,
                    'success': False,
                    'error': str(e),
                    'duration': 0
                })

    return results


def run_experiments_sequential(config_files, timeout_seconds=86400, dataset_type="dummy"):
    """
    Run experiments one by one.

    Args:
        config_files: List of config file paths
        timeout_seconds: Timeout per experiment
        dataset_type: Dataset to use for all experiments
    """
    global completed_count, total_count
    completed_count = 0
    total_count = len(config_files)

    safe_print(f"\n{'#'*70}")
    safe_print(f"# SEQUENTIAL EXECUTION")
    safe_print(f"#   Total experiments: {len(config_files)}")
    safe_print(f"#   Dataset: {dataset_type}")
    safe_print(f"#   Timeout per experiment: {timeout_seconds}s ({timeout_seconds/60:.0f} min)")
    safe_print(f"{'#'*70}\n")

    # List all experiments
    safe_print("Experiments to run:")
    for idx, config in enumerate(config_files, 1):
        info = format_config_info(config)
        safe_print(f"  {idx:2d}. {Path(config).name} ({info})")
    safe_print()

    results = []

    for idx, config_file in enumerate(config_files):
        result = run_single_experiment(
            config_file, idx + 1, len(config_files), timeout_seconds, dataset_type
        )
        results.append(result)

        # Brief pause between experiments
        if idx < len(config_files) - 1:
            remaining = len(config_files) - idx - 1
            safe_print(f"\n⏸️  Cooling down (10s)... {remaining} experiments remaining\n")
            time.sleep(10)

    return results


def save_execution_log(results, output_file='execution_log.json', dataset_type="dummy"):
    """Save execution results for future reference."""

    log_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset_type': dataset_type,
        'total_experiments': len(results),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'total_duration_minutes': sum(r.get('duration', 0) / 60 for r in results),
        'results': results
    }

    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    safe_print(f"\n📄 Execution log saved: {output_path}")

    return log_data


def run_centralized_baseline(timeout_seconds=86400):
    """Run centralized baseline training for comparison."""
    safe_print(f"\n{'='*70}")
    safe_print(f"▶ RUNNING CENTRALIZED BASELINE")
    safe_print(f"{'='*70}")

    simulation_dir = Path(__file__).parent.parent / 'src' / 'simulation'
    cmd = ['python3', 'diloco_trainer.py', '--centralized', '--real-data']

    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd, cwd=str(simulation_dir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        output_lines = []
        while True:
            if time.time() - start_time > timeout_seconds:
                process.kill()
                break
            line = process.stdout.readline()
            if line:
                output_lines.append(line)
                print(line, end='', flush=True)
            elif process.poll() is not None:
                break

        duration = time.time() - start_time
        success = process.returncode == 0
        safe_print(f"\n{'='*70}")
        if success:
            safe_print(f"✅ CENTRALIZED BASELINE COMPLETED ({duration:.1f}s)")
        else:
            safe_print(f"❌ CENTRALIZED BASELINE FAILED")
        safe_print(f"{'='*70}")
        return {'config': 'centralized_baseline', 'success': success,
                'duration': duration, 'duration_minutes': duration / 60}
    except Exception as e:
        safe_print(f"❌ Centralized baseline error: {e}")
        return {'config': 'centralized_baseline', 'success': False,
                'error': str(e), 'duration': time.time() - start_time}


def print_summary(results):
    """Print execution summary."""

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success'] and not r.get('skipped', False)]
    timeout_skipped = [r for r in results if r.get('skipped', False)]

    total_duration = sum(r.get('duration', 0) for r in successful)
    avg_duration = total_duration / len(successful) if successful else 0

    safe_print(f"\n{'#'*70}")
    safe_print(f"# EXECUTION SUMMARY")
    safe_print(f"{'#'*70}")
    safe_print(f"  Total experiments:    {len(results)}")
    safe_print(f"  Successful:           {len(successful)} ✅")
    safe_print(f"  Failed:               {len(failed)} ❌")
    safe_print(f"  Timeout (skipped):    {len(timeout_skipped)} ⏱️")
    safe_print(f"  Total time:           {total_duration/60:.1f} min ({total_duration/86400:.2f} hours)")
    safe_print(f"  Avg time/experiment:  {avg_duration/60:.1f} min")

    if successful:
        safe_print(f"\n✅ Successful experiments ({len(successful)}):")
        for r in successful:
            safe_print(f"   • {r['config']} ({r.get('config_info', '?')}) - {r.get('duration_minutes', 0):.1f} min")

    if failed:
        safe_print(f"\n❌ Failed experiments ({len(failed)}):")
        for r in failed:
            error = r.get('error', 'Unknown error')[:50]
            safe_print(f"   • {r['config']} ({r.get('config_info', '?')}) - {error}")

    if timeout_skipped:
        safe_print(f"\n⏱️ Timeout experiments ({len(timeout_skipped)}):")
        for r in timeout_skipped:
            safe_print(f"   • {r['config']} ({r.get('config_info', '?')}) - {r.get('duration_minutes', 0):.1f} min")

    safe_print(f"{'#'*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run TaniFi DiLoCo experiments in parallel or sequential mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with dummy data (fast testing)
  python run_experiments.py --dummy-data

  # Run with WeedsGalore real data
  python run_experiments.py --real-data

  # Run in parallel with 4 workers
  python run_experiments.py --workers 4 --real-data

  # Run specific batch of configs
  python run_experiments.py --pattern "config_100f_*.yaml" --real-data

Dataset Configuration:
  The dataset can be selected in two ways:
  1. Command line: --real-data or --dummy-data flags (recommended)
  2. Code: Change DATASET_TYPE variable at top of this file

  Command line flags override the code setting.
        """
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1 for sequential)'
    )
    parser.add_argument(
        '--batch',
        type=str,
        default=None,
        help='Filter configs by batch name (e.g., "batch1")'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='config_*.yaml',
        help='Pattern to match config files (default: config_*.yaml)'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Force sequential execution even if workers > 1'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=86400,
        help='Timeout per experiment in seconds (default: 86400 = 24 hours)'
    )

    # Dataset selection arguments (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        '--real-data',
        action='store_true',
        help='Use WeedsGalore real agricultural dataset'
    )
    data_group.add_argument(
        '--dummy-data',
        action='store_true',
        help='Use dummy random data for testing (default)'
    )
    data_group.add_argument(
        '--dataset',
        type=str,
        choices=SUPPORTED_DATASETS,
        default=None,
        help=f'Specify dataset type: {SUPPORTED_DATASETS}'
    )

    parser.add_argument(
        '--run-baseline',
        action='store_true',
        help='Run centralized baseline before federated experiments'
    )

    args = parser.parse_args()

    # Determine dataset type
    # Priority: --dataset > --real-data/--dummy-data > DATASET_TYPE global
    if args.dataset:
        dataset_type = args.dataset
    elif args.real_data:
        dataset_type = "weedsgalore"
    elif args.dummy_data:
        dataset_type = "dummy"
    else:
        dataset_type = DATASET_TYPE  # Use global config

    safe_print(f"\n📊 Dataset configuration: {dataset_type}")

    # Find config files
    config_dir = Path('.')
    config_files = sorted(config_dir.glob(args.pattern))

    # Filter by batch if specified
    if args.batch:
        config_files = [c for c in config_files if args.batch in c.name]
        safe_print(f"🔍 Filtered to batch '{args.batch}': {len(config_files)} configs")

    if not config_files:
        safe_print("❌ No config files found matching pattern")
        safe_print(f"   Pattern: {args.pattern}")
        safe_print(f"   Directory: {config_dir.absolute()}")
        return

    safe_print(f"\n📋 Found {len(config_files)} config files")

    # Global results list for signal handler
    results = []

    def signal_handler(sig, frame):
        """Save partial results on interrupt."""
        safe_print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        if results:
            safe_print("💾 Saving partial results...")
            save_execution_log(results, 'execution_log_partial.json')
            print_summary(results)
        safe_print("👋 Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run centralized baseline first if requested
    start_time = time.time()

    if args.run_baseline:
        baseline_result = run_centralized_baseline(args.timeout)
        results.append(baseline_result)

    # Run federated experiments
    if args.sequential or args.workers == 1:
        results.extend(run_experiments_sequential(config_files, args.timeout, dataset_type))
    else:
        results.extend(run_experiments_parallel(config_files, args.workers, args.timeout, dataset_type))

    total_time = time.time() - start_time

    # Save execution log
    save_execution_log(results, dataset_type=dataset_type)

    # Print summary
    print_summary(results)

    safe_print(f"✅ All experiments completed in {total_time/60:.1f} minutes!")
    safe_print(f"   Check execution_log.json for detailed results")


if __name__ == '__main__':
    main()
