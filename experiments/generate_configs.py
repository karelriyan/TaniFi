#!/usr/bin/env python3
"""
TaniFi Experiment Configuration Generator

This script generates multiple YAML configuration files for DiLoCo federated
learning experiments with varying hyperparameters. It enables systematic
parameter sweeps for research reproducibility.

The generator creates configs for all combinations of:
- Number of farmers (federated clients)
- Number of federated rounds
- Local training steps per round

Usage:
    # Generate configs using parameters defined in CONFIGURATION section
    python generate_configs.py

    # Interactive mode - input parameters manually
    python generate_configs.py --custom

Output:
    Config files are saved as: config_{farmers}f_{rounds}r_{steps}s.yaml
    Example: config_100f_15r_500s.yaml (100 farmers, 15 rounds, 500 steps)

Requirements:
    - PyYAML: pip install pyyaml
    - Base config file (config.yaml) must exist in the same directory

Author: TaniFi Research Team
"""

import yaml
from pathlib import Path
from itertools import product
from datetime import datetime


# ============================================================================
# CONFIGURATION - Modify experiment parameters here
# ============================================================================

# Number of farmers (federated clients) to test
# More farmers = more diverse data distribution, potentially better generalization
FARMERS = [100]

# Number of federated communication rounds
# Higher rounds = more synchronization, better convergence but more communication
ROUNDS = 10

# Local training steps per round before synchronization
# Higher steps = more local computation, less communication (DiLoCo advantage)
STEPS = [300]

# ============================================================================
# END CONFIGURATION
# ============================================================================


def generate_config_files(
    base_config_path: str = 'config.yaml',
    output_dir: str = '.',
    farmers_list: list = None,
    num_rounds: int = None,
    steps_list: list = None,
    verbose: bool = True
) -> int:
    """
    Generate multiple configuration files for parameter sweep experiments.

    This function creates YAML config files for all combinations of the
    specified parameters, enabling systematic hyperparameter exploration.

    Args:
        base_config_path: Path to the base configuration template.
                         All generated configs inherit from this file.
        output_dir: Directory where generated configs will be saved.
                   Created automatically if it doesn't exist.
        farmers_list: List of farmer counts to test. Each value represents
                     the number of federated clients in the simulation.
                     Default: uses FARMERS from CONFIGURATION section.
        num_rounds: Number of federated rounds for training.
                   Default: uses ROUNDS from CONFIGURATION section.
        steps_list: List of local training steps to test. Each value
                   represents steps before federated synchronization.
                   Default: uses STEPS from CONFIGURATION section.
        verbose: If True, print progress information.

    Returns:
        Number of configuration files generated.

    Raises:
        FileNotFoundError: If base_config_path doesn't exist.

    Example:
        >>> generate_config_files(
        ...     farmers_list=[50, 100],
        ...     num_rounds=10,
        ...     steps_list=[100, 500]
        ... )
        # Generates 4 configs: 50f_10r_100s, 50f_10r_500s, 100f_10r_100s, 100f_10r_500s
    """

    # Use defaults from CONFIGURATION section if not provided
    if farmers_list is None:
        farmers_list = FARMERS
    if num_rounds is None:
        num_rounds = ROUNDS
    if steps_list is None:
        steps_list = STEPS

    # Validate inputs
    if not farmers_list:
        raise ValueError("farmers_list cannot be empty")
    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive")
    if not steps_list:
        raise ValueError("steps_list cannot be empty")

    # Load base configuration
    base_config_file = Path(base_config_path)

    if not base_config_file.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    with open(base_config_file, 'r') as f:
        base_config = yaml.safe_load(f)

    # Calculate total configurations
    total_configs = len(farmers_list) * len(steps_list)

    if verbose:
        print(f"{'='*60}")
        print(f"TaniFi Configuration Generator")
        print(f"{'='*60}")
        print(f"")
        print(f"Base config:     {base_config_path}")
        print(f"Output dir:      {output_dir}")
        print(f"")
        print(f"Parameters:")
        print(f"  Farmers:       {farmers_list}")
        print(f"  Rounds:        {num_rounds}")
        print(f"  Local steps:   {steps_list}")
        print(f"")
        print(f"Total configs:   {total_configs}")
        print(f"{'='*60}")
        print(f"")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_count = 0

    # Generate all parameter combinations
    for farmers, steps in product(farmers_list, steps_list):
        # Deep copy base config to avoid mutation
        config = yaml.safe_load(yaml.dump(base_config))

        # Ensure federated section exists
        if 'federated' not in config:
            config['federated'] = {}

        # Update federated learning parameters
        config['federated']['num_farmers'] = farmers
        config['federated']['num_rounds'] = num_rounds
        config['federated']['local_steps'] = steps

        # Remove deprecated checkpoint_rounds if present
        config['federated'].pop('checkpoint_rounds', None)

        # Generate descriptive filename
        filename = f"config_{farmers}f_{num_rounds}r_{steps}s.yaml"
        output_file = output_path / filename

        # Write configuration file
        with open(output_file, 'w') as f:
            # Add header comment for traceability
            f.write(f"# Auto-generated config for TaniFi DiLoCo experiment\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Parameters: {farmers} farmers, {num_rounds} rounds, {steps} local steps\n")
            f.write(f"#\n")
            f.write(f"# To run this experiment:\n")
            f.write(f"#   python run_diloco.py --config {filename}\n")
            f.write(f"\n")
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        generated_count += 1

        if verbose:
            print(f"  [+] {filename}")

    if verbose:
        print(f"")
        print(f"{'='*60}")
        print(f"Successfully generated {generated_count} configuration files")
        print(f"Location: {output_path.absolute()}")
        print(f"{'='*60}")

    return generated_count


def generate_custom_configs() -> int:
    """
    Interactive mode for generating configurations with user input.

    Prompts the user for experiment parameters and generates the
    corresponding configuration files.

    Returns:
        Number of configuration files generated.
    """

    print("="*60)
    print("TaniFi Config Generator - Interactive Mode")
    print("="*60)
    print("")
    print("Enter experiment parameters below.")
    print("For multiple values, separate with commas (e.g., 50,100,150)")
    print("")

    # Get number of farmers
    farmers_input = input("Number of farmers [default: 50,100,150]: ").strip()
    if farmers_input:
        farmers = [int(x.strip()) for x in farmers_input.split(',')]
    else:
        farmers = [50, 100, 150]

    # Get number of rounds
    rounds_input = input("Number of rounds [default: 15]: ").strip()
    if rounds_input:
        rounds = int(rounds_input)
    else:
        rounds = 15

    # Get local steps
    steps_input = input("Local steps [default: 100,300,500]: ").strip()
    if steps_input:
        steps = [int(x.strip()) for x in steps_input.split(',')]
    else:
        steps = [100, 300, 500]

    print("")

    return generate_config_files(
        base_config_path='config.yaml',
        output_dir='.',
        farmers_list=farmers,
        num_rounds=rounds,
        steps_list=steps
    )


def main():
    """Main entry point for the configuration generator."""

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--custom':
        count = generate_custom_configs()
    else:
        count = generate_config_files()

    # Print next steps
    print("")
    print("Next steps:")
    print("  1. Review generated configs in the experiments/ directory")
    print("  2. Run experiments: python run_experiments.py")
    print("  3. Analyze results: see notebooks/analysis_template.ipynb")
    print("")

    return count


if __name__ == '__main__':
    main()
