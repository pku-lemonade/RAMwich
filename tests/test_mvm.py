import logging

import numpy as np

from ramwich import RAMwich

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    config_file = "examples/mlp_l4_mnist/config.yaml"
    ops_file = "examples/mlp_l4_mnist/ops.json"
    weights_file = "examples/mlp_l4_mnist/weights.npz"

    simulator = RAMwich(config_file=config_file)
    simulator.load_operations(file_path=ops_file)
    simulator.load_weights(file_path=weights_file)

    core = simulator.get_node(0).get_tile(2).get_core(0)

    input_vec = np.random.randint(0, 2**15 - 1, size=(core.config.mvmu_config.xbar_config.xbar_size,), dtype=np.int32)
    weights = np.load(weights_file)
    matrix = weights["node0_tile2_core0_mvmu0"].astype(np.float64)

    core.write_to_register(0, input_vec)
    core.get_mvmu(0).execute_mvm()
    output = core.read_from_register(
        core.config.num_mvmus_per_core * core.config.mvmu_config.xbar_config.xbar_size,
        core.config.mvmu_config.xbar_config.xbar_size,
    ) * (2**-8)
    expected_output = np.dot(matrix, input_vec * (2**-8))
    error_ratio = np.abs((output - expected_output) / expected_output)

    output_precise = core.get_mvmu(0).output_register_array.read() * (2**-16)
    error_ratio_precise = np.abs((output_precise - expected_output) / expected_output)

    print(error_ratio)
    print("___________________________________________________")
    print(error_ratio_precise)


if __name__ == "__main__":
    main()
