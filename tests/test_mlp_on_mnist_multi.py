import logging
import time

import numpy as np

from ramwich import RAMwich

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    config_file = "examples/mlp_l4_mnist/config.yaml"
    ops_file = "examples/mlp_l4_mnist/ops.json"
    weights_file = "examples/mlp_l4_mnist/weights.npz"
    activation_file = "examples/mlp_l4_mnist/mnist_test.npy"
    label_file = "examples/mlp_l4_mnist/mnist_test_labels.npy"

    activation = np.load(activation_file)
    label = np.load(label_file)

    batches = 1000
    correct = 0

    start_time = time.perf_counter()

    simulator = RAMwich(config_file=config_file, ops_file=ops_file, weights_file=weights_file)

    for test_id in range(1, batches + 1):
        simulator.reset()
        simulator.run(activation=activation[test_id])

        output = simulator.get_node(0).get_tile(1).edram.cells[:10]
        output_float = output.astype(np.float64) / (1 << 8)
        if np.argmax(output_float) == label[test_id]:
            correct += 1
        logger.info(f"Test ID: {test_id}, Correct: {correct}/{test_id}")

    end_time = time.perf_counter()

    accuracy = correct / batches
    logger.info(f"Accuracy: {accuracy:.2%}")
    logger.info(f"Total simulation time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
