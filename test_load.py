import argparse
import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ramwich import RAMwich
from ramwich.utils.visualize import summarize_results

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAMwich Simulator")
    parser.add_argument("--ops", required=True, help="OP file (JSON)")
    parser.add_argument("--config", required=True, help="Configuration file (YAML)")
    args = parser.parse_args()

    simulator = RAMwich(config_file=args.config)
    simulator.load_operations(file_path=args.ops)


if __name__ == "__main__":
    main()