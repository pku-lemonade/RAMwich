import argparse
import logging
import os

from .simulator import RAMwichSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='RAMwich Simulator')
    parser.add_argument('--ops', required=True, help='OP file (JSON)')
    parser.add_argument('--config', required=True, help='Configuration file (YAML)')
    args = parser.parse_args()

    simulator = RAMwichSimulator(config_file=args.config)
    simulator.run(ops_file=args.ops)

if __name__ == "__main__":
    main()
