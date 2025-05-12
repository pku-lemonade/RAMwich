import concurrent.futures
import logging
import time

import numpy as np

from ramwich import RAMwich

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def process_batch(
    worker_id: int, sample_indices: list[int], activation: np.ndarray, labels: np.ndarray, config_dict: dict
) -> list[bool]:
    """Process a batch of samples with a single simulator instance

    Args:
        worker_id: ID of the worker thread
        sample_indices: List of indices to process
        activation: Full activation dataset
        labels: Full label dataset
        config_dict: Dictionary with configuration paths

    Returns:
        List of booleans indicating correct/incorrect predictions
    """
    # Create a single simulator instance for this worker
    simulator = RAMwich(
        config_file=config_dict["config_file"],
        ops_file=config_dict["ops_file"],
        weights_file=config_dict["weights_file"],
    )

    results = []
    for i, test_id in enumerate(sample_indices):
        if i > 0:  # Only reset after the first sample
            simulator.reset()

        # Run simulation for this sample
        simulator.run(activation=activation[test_id])

        # Check prediction
        output = simulator.get_node(0).get_tile(1).edram.cells[:10]
        output_float = output.astype(np.float64) / (1 << 8)
        correct = np.argmax(output_float) == labels[test_id]
        results.append(correct)

        # Log progress (less frequently to reduce log spam)
        if (i + 1) % 10 == 0 or i == 0 or i == len(sample_indices) - 1:
            correct_count = sum(results)
            logger.info(
                f"Worker {worker_id}: Completed {i + 1}/{len(sample_indices)}, "
                f"Batch Accuracy: {correct_count / (i + 1):.2%}"
            )

    return results


def main():
    config_file = "examples/mlp_l4_mnist/config.yaml"
    ops_file = "examples/mlp_l4_mnist/ops.json"
    weights_file = "examples/mlp_l4_mnist/weights.npz"
    activation_file = "examples/mlp_l4_mnist/mnist_test.npy"
    label_file = "examples/mlp_l4_mnist/mnist_test_labels.npy"

    activation = np.load(activation_file)
    label = np.load(label_file)

    batches = 10000
    num_workers = 64  # Adjust based on CPU cores

    start_time = time.perf_counter()

    configs = {
        "config_file": config_file,
        "ops_file": ops_file,
        "weights_file": weights_file,
    }

    # Divide work among workers
    sample_indices = list(range(0, batches))
    samples_per_worker = len(sample_indices) // num_workers
    remainder = len(sample_indices) % num_workers

    batches_per_worker = []
    start_idx = 0
    for i in range(num_workers):
        worker_samples = samples_per_worker + (1 if i < remainder else 0)
        end_idx = start_idx + worker_samples
        batches_per_worker.append(sample_indices[start_idx:end_idx])
        start_idx = end_idx

    # Process each batch in parallel
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_worker = {
            executor.submit(process_batch, worker_id, batch, activation, label, configs): worker_id
            for worker_id, batch in enumerate(batches_per_worker)
        }

        for future in concurrent.futures.as_completed(future_to_worker):
            worker_id = future_to_worker[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.info(f"Worker {worker_id} completed with accuracy: {sum(results) / len(results):.2%}")
            except Exception as exc:
                logger.error(f"Worker {worker_id} failed: {exc}")

    end_time = time.perf_counter()

    correct_count = sum(all_results)
    accuracy = correct_count / len(all_results)
    print(f"workers: {num_workers}, batches: {batches}, samples per worker: {samples_per_worker}")
    print(f"Final Accuracy: {accuracy:.2%}")
    print(f"Total samples: {len(all_results)}, Correct: {correct_count}")
    print(f"Total simulation time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
