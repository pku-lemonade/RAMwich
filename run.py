import argparse
import logging

from ramwich import RAMwich

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAMwich Simulator")
    parser.add_argument("--JSONConfig", required=True, help="Configuration file (JSON)")
    parser.add_argument("--YAMLConfig", required=True, help="Configuration file (YAML)")
    parser.add_argument("--ops", required=True, help="OP file (JSON)")
    parser.add_argument("--params", required=False, help="Parameters file (NPZ)")
    parser.add_argument("--activation", required=False, help="activation file (NPY)")
    parser.add_argument("--timeout", type=int, default=100000, help="Simulation timeout in cycles (default: 100000)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug monitoring")
    args = parser.parse_args()

    simulator = RAMwich(
        json_config_file=args.JSONConfig, yaml_config_file=args.YAMLConfig, ops_file=args.ops, params_file=args.params
    )
    simulator.run(activation=args.activation, timeout=args.timeout, enable_debug_monitor=not args.no_debug)

    # Get statistics and pass to visualization
    # stats = simulator.get_stats()
    # summarize_results(stats)


if __name__ == "__main__":
    main()
