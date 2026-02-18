import argparse
import os, sys
# Ensure the project root is in sys.path for absolute imports when run as a script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # When executed as a module within the package
    from .training import main_training
except ImportError:
    # Fallback for direct script execution using absolute import
    from src.simulation.training import main_training


def parse_args():
    parser = argparse.ArgumentParser(description="TaniFi simulation entry point")
    parser.add_argument("--config", type=str, help="Path to a YAML configuration file")
    parser.add_argument("--centralized", action="store_true", help="Run centralized baseline instead of federated training")
    parser.add_argument("--real-data", action="store_true", help="Use the real WeedsGalore dataset; otherwise synthetic data is used")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false", help="Disable saving training plots")
    parser.set_defaults(save_plots=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_training(
        config_file=args.config,
        centralized=args.centralized,
        real_data=args.real_data,
        save_plots=args.save_plots,
    )
