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
    activation_file = "examples/mlp_l4_mnist/activation.npy"

    simulator = RAMwich(config_file=config_file)
    simulator.run(ops_file=ops_file, weights_file=weights_file, activation=activation_file)

    output = simulator.get_node(0).get_tile(1).edram.cells[:10]
    output_float = output.astype(np.float64) / (1 << 8)
    print(f"Output:{output_float}")
    print(f"Output(Label):{np.argmax(output_float)}")

    print(f"Expected Output(Label):{7}")


if __name__ == "__main__":
    main()
