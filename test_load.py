import argparse
import logging

from ramwich import RAMwich
from ramwich.utils.visualize import summarize_results

from ramwich.ops import Send

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAMwich Simulator")
    parser.add_argument("--ops", required=True, help="OP file (JSON)")
    parser.add_argument("--config", required=True, help="Configuration file (YAML)")
    parser.add_argument("--weight", required=False, help="Weight file (JSON)")
    args = parser.parse_args()

    simulator = RAMwich(config_file=args.config)
    simulator.load_operations(file_path=args.ops)
    simulator.load_weights(file_path=args.weight)

    validate_op = Send(
        type="send",
        node=0,
        tile=0,
        mem_addr=768,
        target_tile=3,
        width=16,
        vec=1
    )
    assert (simulator.get_node(0).get_tile(0).operations[0] == validate_op), "Operation not loaded correctly"


if __name__ == "__main__":
    main()