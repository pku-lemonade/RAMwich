#!/usr/bin/env python3

import numpy as np
import pytest
from ramwich.config import Config
from ramwich.config.data_config import DataConfig, BitConfig
from ramwich.mvmu import MVMU


def test_parallel_cnn_multi_channel_sram_cim():
    """Test parallel CNN with multiple channels using SRAM CIM"""
    # Configuration for parallel CNN processing
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM, BitConfig.SRAM],  # 3-bit weights
        weight_format="Q3.0",
        activation_format="Q8.0"
    )
    
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=data_config
    )
    
    mvmu = MVMU(id=0, config=config)
    
    # Verify SRAM CIM configuration
    assert mvmu.mvmu_config.have_sram_xbar == True
    assert mvmu.mvmu_config.num_sram_xbar_per_mvmu == 3
    
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    
    # Create multi-channel convolution weights
    # Simulate 3x3 conv with multiple input/output channels
    np.random.seed(100)
    weights = np.random.normal(0, 0.2, size=(xbar_size, xbar_size)).astype(np.float64)
    weights = np.clip(weights, -1, 1)  # Clip for 3-bit quantization
    
    mvmu.load_weights(weights)
    
    # Test with multi-channel input (RGB-like)
    input_data = np.random.randint(0, 255, size=xbar_size).astype(np.int32)
    mvmu.write_to_inreg(0, input_data)
    
    mvmu.execute_mvm()
    output = mvmu.read_from_outreg(0, xbar_size)
    
    # Verify output characteristics
    assert output.shape == (xbar_size,)
    assert len(np.unique(output)) > 10, "Multi-channel CNN should produce diverse outputs"
    
    # Verify statistics
    stats = mvmu.get_stats()
    assert "SRAM CIM Unit" in stats
    assert stats["SRAM CIM Unit"].activation_count > 0


def test_parallel_cnn_depthwise_separable():
    """Test depthwise separable convolution with SRAM CIM"""
    # Configuration for depthwise separable conv (DS-CNN style)
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM],  # 2-bit for efficiency
        weight_format="Q2.0",
        activation_format="Q8.0"
    )
    
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=data_config
    )
    
    mvmu = MVMU(id=0, config=config)
    
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    
    # Create depthwise convolution weights (sparse pattern)
    weights = np.zeros((xbar_size, xbar_size), dtype=np.float64)
    
    # Simulate depthwise pattern - each channel processes independently
    np.random.seed(200)
    for i in range(0, min(xbar_size, 64), 8):  # Every 8th element for different channels
        for j in range(min(8, xbar_size - i)):  # 8 weights per channel
            if i + j < xbar_size:
                weights[i + j, i + j] = np.random.choice([-1, 0, 1])  # Ternary weights
    
    mvmu.load_weights(weights)
    
    # Test with channel-separated input
    input_data = np.zeros(xbar_size, dtype=np.int32)
    for i in range(0, min(xbar_size, 64), 8):
        for j in range(min(8, xbar_size - i)):
            if i + j < xbar_size:
                input_data[i + j] = np.random.randint(0, 128)
    
    mvmu.write_to_inreg(0, input_data)
    mvmu.execute_mvm()
    output = mvmu.read_from_outreg(0, xbar_size)
    
    # Verify depthwise processing
    assert output.shape == (xbar_size,)
    
    # Check that non-zero inputs produce non-zero outputs (for non-zero weights)
    non_zero_input_indices = np.nonzero(input_data)[0]
    if len(non_zero_input_indices) > 0:
        # At least some outputs should be non-zero
        assert np.count_nonzero(output) > 0, "Depthwise conv should produce some non-zero outputs"
    
    # Verify statistics
    stats = mvmu.get_stats()
    assert "SRAM CIM Unit" in stats
    assert stats["SRAM CIM Unit"].activation_count > 0


def test_ds_cnn_pointwise_convolution():
    """Test DS-CNN pointwise (1x1) convolution with SRAM CIM"""
    # Configuration for pointwise convolution
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM] * 4,  # 4-bit for pointwise precision
        weight_format="Q4.0",
        activation_format="Q8.0"
    )
    
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=data_config
    )
    
    mvmu = MVMU(id=0, config=config)
    
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    
    # Create pointwise convolution weights (dense, 1x1 conv)
    np.random.seed(300)
    weights = np.random.normal(0, 0.3, size=(xbar_size, xbar_size)).astype(np.float64)
    weights = np.clip(weights, -1, 1)  # Clip for 4-bit quantization
    
    mvmu.load_weights(weights)
    
    # Test with feature map input (after depthwise conv)
    input_data = np.random.randint(-64, 64, size=xbar_size).astype(np.int32)
    mvmu.write_to_inreg(0, input_data)
    
    mvmu.execute_mvm()
    output = mvmu.read_from_outreg(0, xbar_size)
    
    # Verify pointwise convolution characteristics
    assert output.shape == (xbar_size,)
    
    # Pointwise conv should mix channels, creating diverse outputs
    output_std = np.std(output.astype(float))
    input_std = np.std(input_data.astype(float))
    assert output_std > 0, "Pointwise conv should produce varied outputs"
    
    # Verify statistics
    stats = mvmu.get_stats()
    assert "SRAM CIM Unit" in stats
    assert stats["SRAM CIM Unit"].activation_count > 0


def test_parallel_cnn_batch_processing():
    """Test parallel CNN with batch processing simulation"""
    # Configuration for batch processing
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM, BitConfig.SRAM],
        weight_format="Q3.0",
        activation_format="Q8.0"
    )
    
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=data_config
    )
    
    mvmu = MVMU(id=0, config=config)
    
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    
    # Create shared convolution weights
    np.random.seed(400)
    weights = np.random.normal(0, 0.25, size=(xbar_size, xbar_size)).astype(np.float64)
    weights = np.clip(weights, -1, 1)
    
    mvmu.load_weights(weights)
    
    # Simulate batch processing by running multiple inputs
    batch_size = 4
    batch_outputs = []
    
    for batch_idx in range(batch_size):
        # Different input for each batch item
        np.random.seed(400 + batch_idx)
        input_data = np.random.randint(0, 200, size=xbar_size).astype(np.int32)
        
        mvmu.write_to_inreg(0, input_data)
        mvmu.execute_mvm()
        output = mvmu.read_from_outreg(0, xbar_size)
        
        batch_outputs.append(output)
        
        # Reset for next batch item (in real hardware, this would be automatic)
        mvmu.output_register_array.clean_cells()
    
    # Verify batch processing results
    assert len(batch_outputs) == batch_size
    
    # Each batch item should produce different outputs
    for i in range(batch_size):
        assert batch_outputs[i].shape == (xbar_size,)
        for j in range(i + 1, batch_size):
            # Outputs should be different for different inputs
            assert not np.array_equal(batch_outputs[i], batch_outputs[j]), \
                f"Batch items {i} and {j} should produce different outputs"
    
    # Verify statistics accumulated across batch
    stats = mvmu.get_stats()
    assert "SRAM CIM Unit" in stats
    assert stats["SRAM CIM Unit"].activation_count >= batch_size, \
        "Statistics should accumulate across batch processing"


def test_ds_cnn_mobile_optimization():
    """Test DS-CNN mobile optimization features with SRAM CIM"""
    # Low-precision configuration for mobile deployment
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM],  # 2-bit for mobile efficiency
        weight_format="Q2.0",
        activation_format="Q8.0"
    )
    
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=data_config
    )
    
    mvmu = MVMU(id=0, config=config)
    
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    
    # Create mobile-optimized weights (sparse, ternary)
    weights = np.zeros((xbar_size, xbar_size), dtype=np.float64)
    
    # Create sparse ternary weights for mobile efficiency
    np.random.seed(500)
    sparsity = 0.7  # 70% sparse for mobile efficiency
    for i in range(xbar_size):
        for j in range(xbar_size):
            if np.random.random() > sparsity:  # Only 30% non-zero
                weights[i, j] = np.random.choice([-1, 0, 1])  # Ternary
    
    mvmu.load_weights(weights)
    
    # Test with mobile-typical input (lower precision)
    input_data = np.random.randint(0, 64, size=xbar_size).astype(np.int32)  # Reduced range
    mvmu.write_to_inreg(0, input_data)
    
    mvmu.execute_mvm()
    output = mvmu.read_from_outreg(0, xbar_size)
    
    # Verify mobile optimization characteristics
    assert output.shape == (xbar_size,)
    
    # With sparse weights, many outputs might be zero, but some should be non-zero
    non_zero_outputs = np.count_nonzero(output)
    total_outputs = len(output)
    sparsity_ratio = 1 - (non_zero_outputs / total_outputs)
    
    # Should have some sparsity but not complete sparsity
    assert 0 < sparsity_ratio < 1, f"Output sparsity should be reasonable: {sparsity_ratio}"
    
    # Verify energy efficiency statistics
    stats = mvmu.get_stats()
    assert "SRAM CIM Unit" in stats
    assert stats["SRAM CIM Unit"].activation_count > 0
    
    # For mobile optimization, energy should be reasonable
    energy_per_op = stats["SRAM CIM Unit"].dynamic_energy / max(1, stats["SRAM CIM Unit"].activation_count)
    assert energy_per_op >= 0, "Energy per operation should be non-negative"


if __name__ == "__main__":
    test_parallel_cnn_multi_channel_sram_cim()
    test_parallel_cnn_depthwise_separable()
    test_ds_cnn_pointwise_convolution()
    test_parallel_cnn_batch_processing()
    test_ds_cnn_mobile_optimization()
    print("All Parallel CNN and DS-CNN SRAM CIM tests passed!")