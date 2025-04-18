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
    parser.add_argument("--weights", required=False, help="Weight file (JSON)")
    args = parser.parse_args()

    simulator = RAMwich(config_file=args.config)
    simulator.load_operations(file_path=args.ops)
    simulator.load_weights(file_path=args.weights)

    mvmu = simulator.get_node(0).get_tile(2).get_core(0).mvmus[0]

    input_vec = np.random.randint(0, 2**15 - 1, size=(mvmu.mvmu_config.xbar_config.xbar_size,), dtype=np.int16)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_path = os.path.join(script_dir, "mvm_test_matrix.npy")
    matrix = np.load(matrix_path)

    mvmu.write_to_inreg(0, input_vec)
    mvmu.execute_mvm()
    output = mvmu.read_from_outreg(0, mvmu.mvmu_config.xbar_config.xbar_size) * (2**-8)
    expected_output = np.dot(matrix, input_vec * (2**-8))
    error = output - expected_output

    print(error)


if __name__ == "__main__":
    main()
