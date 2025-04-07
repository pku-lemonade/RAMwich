import argparse
import logging

from .ramwich import RAMwich
from .utils.visualize import summarize_results

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAMwich Simulator")
    parser.add_argument("--ops", required=True, help="OP file (JSON)")
    parser.add_argument("--config", required=True, help="Configuration file (YAML)")
    args = parser.parse_args()

    simulator = RAMwich(config_file=args.config)
    simulator.run(ops_file=args.ops)

    # Get statistics and pass to visualization
    stats = simulator.get_stats()
    summarize_results(stats)


if __name__ == "__main__":
    main()
