#!/usr/bin/env python3

import numpy as np
import pytest
from ramwich.config import Config
from ramwich.config.data_config import DataConfig, BitConfig
from ramwich.mvmu import MVMU


def test_lenet5_conv_layer_sram_cim():
    """Test LeNet-5 style convolutional layer with SRAM CIM"""
    # Create SRAM CIM configuration for LeNet-5
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM],  # 2-bit weights
        weight_format="Q2.0",  # 2 bits, no fractional
        activation_format="Q8.0"  # 8 bits, no fractional
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
    assert mvmu.mvmu_config.num_sram_xbar_per_mvmu == 2
    
    # Test with LeNet-5 style 5x5 convolution kernel (flattened for MVM)
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    
    # Create simple convolution-like weights (edge detection kernel pattern)
    weights = np.zeros((xbar_size, xbar_size), dtype=np.float64)
    
    # Simple edge detection pattern in first few rows
    edge_kernel = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  8, -1,  8, -1],
        [-1, -1, -1, -1, -1],
        [-1,  8, -1,  8, -1],
        [-1, -1, -1, -1, -1]
    ]).flatten()
    
    # Place the kernel pattern in the weight matrix
    for i in range(min(25, xbar_size)):
        if i < len(edge_kernel):
            weights[i, i] = edge_kernel[i] / 8.0  # Normalize to [-1, 1] range
    
    mvmu.load_weights(weights)
    
    # Test with image-like input data
    input_data = np.random.randint(0, 256, size=xbar_size).astype(np.int32)
    mvmu.write_to_inreg(0, input_data)
    
    mvmu.execute_mvm()
    output = mvmu.read_from_outreg(0, xbar_size)
    
    # Verify output shape and basic functionality
    assert output.shape == (xbar_size,)
    assert not np.all(output == 0), "Output should not be all zeros"
    
    # Verify statistics are collected
    stats = mvmu.get_stats()
    assert "SRAM CIM Unit" in stats
    assert stats["SRAM CIM Unit"].activation_count > 0


def test_lenet5_fc_layer_sram_cim():
    """Test LeNet-5 style fully connected layer with SRAM CIM"""
    # Create SRAM CIM configuration for FC layer
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM, BitConfig.SRAM],  # 3-bit weights
        weight_format="Q3.0",  # 3 bits, no fractional
        activation_format="Q8.0"  # 8 bits, no fractional
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
    
    # Create random weights for FC layer (typical for LeNet-5 classification)
    np.random.seed(42)  # For reproducible results
    weights = np.random.uniform(-1, 1, size=(xbar_size, xbar_size)).astype(np.float64)
    
    mvmu.load_weights(weights)
    
    # Test with feature vector input (typical FC layer input)
    input_data = np.random.randint(0, 128, size=xbar_size).astype(np.int32)
    mvmu.write_to_inreg(0, input_data)
    
    mvmu.execute_mvm()
    output = mvmu.read_from_outreg(0, xbar_size)
    
    # Verify output shape and basic functionality
    assert output.shape == (xbar_size,)
    
    # For FC layer, we expect varied outputs (not all same value)
    assert len(np.unique(output)) > 1, "FC layer should produce varied outputs"
    
    # Verify statistics are collected
    stats = mvmu.get_stats()
    assert "SRAM CIM Unit" in stats
    assert stats["SRAM CIM Unit"].activation_count > 0


def test_lenet5_mixed_precision_sram_cim():
    """Test LeNet-5 with mixed SRAM CIM and RRAM configuration"""
    # Mixed configuration: SRAM for low precision, RRAM for higher precision
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM, BitConfig.MLC, BitConfig.MLC],
        weight_format="Q0.6",  # 2 SRAM bits + 4 RRAM bits = 6 bits total
        activation_format="Q8.0"
    )
    
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=data_config
    )
    
    mvmu = MVMU(id=0, config=config)
    
    # Verify mixed configuration
    assert mvmu.mvmu_config.have_sram_xbar == True
    assert mvmu.mvmu_config.have_rram_xbar == True
    assert mvmu.mvmu_config.num_sram_xbar_per_mvmu == 2
    assert mvmu.mvmu_config.num_rram_xbar_per_mvmu == 2
    
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    
    # Create weights with higher precision for mixed configuration
    weights = np.random.uniform(-0.5, 0.5, size=(xbar_size, xbar_size)).astype(np.float64)
    
    mvmu.load_weights(weights)
    
    # Test with typical neural network input
    input_data = np.random.randint(0, 64, size=xbar_size).astype(np.int32)
    mvmu.write_to_inreg(0, input_data)
    
    mvmu.execute_mvm()
    output = mvmu.read_from_outreg(0, xbar_size)
    
    # Verify output shape and functionality
    assert output.shape == (xbar_size,)
    
    # Verify both SRAM and RRAM stats are collected
    stats = mvmu.get_stats()
    assert "SRAM CIM Unit" in stats
    assert "RRAM Xbar" in stats
    assert stats["SRAM CIM Unit"].activation_count > 0


if __name__ == "__main__":
    test_lenet5_conv_layer_sram_cim()
    test_lenet5_fc_layer_sram_cim()
    test_lenet5_mixed_precision_sram_cim()
    print("All LeNet-5 SRAM CIM tests passed!")