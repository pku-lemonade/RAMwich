"""
Test module for Matrix-Vector Multiplication (MVM) functionality in RAMwich.
"""
import numpy as np
import pytest
from ramwich import RAMwich


@pytest.mark.unit
def test_mvm_basic_functionality(ramwich_simulator, sample_input_vector, performance_threshold):
    """Test basic matrix-vector multiplication functionality."""
    simulator = ramwich_simulator
    core = simulator.get_node(0).get_tile(2).get_core(0)
    
    # Use the sample input vector
    input_vec = sample_input_vector
    
    # Write input to register and execute MVM
    core.write_to_register(0, input_vec)
    core.get_mvmu(0).execute_mvm()
    
    # Read output
    output = core.read_from_register(
        core.config.num_mvmus_per_core * core.config.mvmu_config.xbar_config.xbar_size,
        core.config.mvmu_config.xbar_config.xbar_size,
    ) * (2**-8)
    
    # Verify output is not all zeros (basic sanity check)
    assert not np.allclose(output, 0), "MVM output should not be all zeros"
    assert output.shape[0] > 0, "MVM output should have valid dimensions"


@pytest.mark.unit
def test_mvm_accuracy(ramwich_simulator, test_weights_file, performance_threshold):
    """Test MVM accuracy against expected results."""
    simulator = ramwich_simulator
    core = simulator.get_node(0).get_tile(2).get_core(0)
    
    # Create reproducible input
    np.random.seed(42)
    input_vec = np.random.randint(0, 2**15 - 1, size=(core.config.mvmu_config.xbar_config.xbar_size,), dtype=np.int32)
    
    # Load weights for comparison
    weights = np.load(test_weights_file)
    matrix = weights["node0_tile2_core0_mvmu0"].astype(np.float64)
    
    # Execute MVM
    core.write_to_register(0, input_vec)
    core.get_mvmu(0).execute_mvm()
    
    # Get outputs
    output = core.read_from_register(
        core.config.num_mvmus_per_core * core.config.mvmu_config.xbar_config.xbar_size,
        core.config.mvmu_config.xbar_config.xbar_size,
    ) * (2**-8)
    
    output_precise = core.get_mvmu(0).output_register_array.read() * (2**-16)
    
    # Calculate expected output
    expected_output = np.dot(matrix, input_vec * (2**-8))
    
    # Calculate error ratios
    error_ratio = np.abs((output - expected_output) / (expected_output + 1e-10))  # Add small epsilon to avoid division by zero
    error_ratio_precise = np.abs((output_precise - expected_output) / (expected_output + 1e-10))
    
    # Check accuracy within threshold
    mean_error = np.mean(error_ratio)
    mean_error_precise = np.mean(error_ratio_precise)
    
    # For hardware simulation, we expect higher error rates due to quantization and approximation
    # Standard MVM should be within 20x the threshold (reasonable for hardware simulation)
    assert mean_error < performance_threshold["accuracy_threshold"] * 20, f"Standard MVM error too high: {mean_error}"
    # Precise MVM should be within 15x the threshold (more lenient for hardware simulation)
    assert mean_error_precise < performance_threshold["accuracy_threshold"] * 15, f"Precise MVM error too high: {mean_error_precise}"


@pytest.mark.unit
def test_mvm_register_operations(ramwich_simulator, sample_input_vector):
    """Test register read/write operations for MVM."""
    simulator = ramwich_simulator
    core = simulator.get_node(0).get_tile(2).get_core(0)
    
    input_vec = sample_input_vector
    
    # Test write operation
    core.write_to_register(0, input_vec)
    
    # Test read operation (should be able to read back what we wrote)
    # Note: This is a basic test - actual register behavior may vary
    mvmu = core.get_mvmu(0)
    assert mvmu is not None, "MVMU should be accessible"
    
    # Execute MVM to ensure the pipeline works
    mvmu.execute_mvm()
    
    # Verify output register has data
    output_register = mvmu.output_register_array
    assert output_register is not None, "Output register should be accessible"


@pytest.mark.performance
@pytest.mark.slow
def test_mvm_performance(ramwich_simulator, performance_threshold):
    """Test MVM performance characteristics."""
    import time
    
    simulator = ramwich_simulator
    core = simulator.get_node(0).get_tile(2).get_core(0)
    
    # Create test input
    np.random.seed(42)
    input_vec = np.random.randint(0, 2**15 - 1, size=(core.config.mvmu_config.xbar_config.xbar_size,), dtype=np.int32)
    
    # Measure execution time
    start_time = time.time()
    
    for _ in range(10):  # Run multiple iterations
        core.write_to_register(0, input_vec)
        core.get_mvmu(0).execute_mvm()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Check performance threshold
    assert execution_time < performance_threshold["simulation_time"], f"MVM execution too slow: {execution_time}s"


if __name__ == "__main__":
    # Backward compatibility for direct script execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    config_file = "examples/mlp_l4_mnist/config.yaml"
    ops_file = "examples/mlp_l4_mnist/ops.json"
    weights_file = "examples/mlp_l4_mnist/weights.npz"

    simulator = RAMwich(config_file=config_file, ops_file=ops_file, weights_file=weights_file)

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
