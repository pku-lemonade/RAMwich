import argparse
import logging
import os

import numpy as np

from ramwich import RAMwich
from ramwich.ops import Send
from ramwich.utils.visualize import summarize_results

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAMwich Simulator")
    parser.add_argument("--ops", required=True, help="OP file (JSON)")
    parser.add_argument("--config", required=True, help="Configuration file (YAML)")
    parser.add_argument("--weights", required=False, help="Weight file (NPZ)")
    args = parser.parse_args()

    simulator = RAMwich(config_file=args.config)
    simulator.load_operations(file_path=args.ops)
    simulator.load_weights(file_path=args.weights)

    core = simulator.get_node(0).get_tile(2).get_core(0)

    input_vec = np.random.randint(0, 2**15 - 1, size=(core.config.mvmu_config.xbar_config.xbar_size,), dtype=np.int32)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights = np.load(args.weights)
    matrix = weights["node0_tile2_core0_mvmu0"].astype(np.float64)

    core.write_to_register(0, input_vec)
    core.get_mvmu(0).execute_mvm()
    output = core.read_from_register(
        core.config.num_mvmus_per_core * core.config.mvmu_config.xbar_config.xbar_size,
        core.config.mvmu_config.xbar_config.xbar_size,
    ) * (2**-8)
    expected_output = np.dot(matrix, input_vec * (2**-8))
    error_ratio = np.abs((output - expected_output) / expected_output)

    print(error_ratio)


if __name__ == "__main__":
    main()
