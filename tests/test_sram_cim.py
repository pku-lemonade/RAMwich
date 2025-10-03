#!/usr/bin/env python3

import numpy as np
import pytest
from ramwich.config import Config
from ramwich.mvmu import MVMU


def test_sram_cim_basic_functionality():
    """Test basic SRAM CIM functionality"""
    # Create data config with SRAM CIM enabled
    from ramwich.config.data_config import DataConfig, BitConfig
    
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM, BitConfig.SRAM, BitConfig.SRAM],
        weight_format="Q0.4",  # 4 bits total to match 4 SRAM units
        activation_format="Q8.8"
    )
    
    # Create configuration with SRAM CIM enabled
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=data_config
    )
    
    # Create MVMU
    mvmu = MVMU(id=0, config=config)
    
    # Verify SRAM CIM is enabled
    assert mvmu.mvmu_config.have_sram_xbar == True
    assert mvmu.mvmu_config.have_rram_xbar == False
    assert mvmu.mvmu_config.num_sram_xbar_per_mvmu == 4
    assert mvmu.mvmu_config.num_rram_xbar_per_mvmu == 0
    
    # Test weight loading
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    weights = np.random.randint(-1, 2, size=(xbar_size, xbar_size)).astype(np.float64)
    
    mvmu.load_weights(weights)
    
    # Test matrix-vector multiplication
    input_data = np.random.randint(0, 8, size=xbar_size).astype(np.int32)
    mvmu.write_to_inreg(0, input_data)
    
    # Execute MVM
    mvmu.execute_mvm()
    
    # Read results
    output = mvmu.read_from_outreg(0, xbar_size)
    
    # Verify output shape
    assert output.shape == (xbar_size,)
    
    # Verify statistics are collected
    stats = mvmu.get_stats()
    assert "SRAM CIM Unit" in stats
    assert stats["SRAM CIM Unit"].activation_count > 0


def test_sram_cim_mixed_configuration():
    """Test mixed SRAM CIM and RRAM configuration"""
    from ramwich.config.data_config import DataConfig, BitConfig
    
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM, BitConfig.MLC, BitConfig.MLC],
        weight_format="Q0.6",  # 2 SRAM (2 bits) + 2 MLC (4 bits) = 6 bits total
        activation_format="Q8.8"
    )
    
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=data_config
    )
    
    mvmu = MVMU(id=0, config=config)
    
    # Verify both are enabled
    assert mvmu.mvmu_config.have_sram_xbar == True
    assert mvmu.mvmu_config.have_rram_xbar == True
    assert mvmu.mvmu_config.num_sram_xbar_per_mvmu == 2
    assert mvmu.mvmu_config.num_rram_xbar_per_mvmu == 2
    
    # Test weight loading and execution
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    weights = np.random.randint(-1, 2, size=(xbar_size, xbar_size)).astype(np.float64)
    mvmu.load_weights(weights)
    
    input_data = np.random.randint(0, 8, size=xbar_size).astype(np.int32)
    mvmu.write_to_inreg(0, input_data)
    
    mvmu.execute_mvm()
    output = mvmu.read_from_outreg(0, xbar_size)
    
    assert output.shape == (xbar_size,)
    
    # Verify both SRAM and RRAM stats are collected
    stats = mvmu.get_stats()
    assert "SRAM CIM Unit" in stats
    assert "RRAM Xbar" in stats


def test_sram_cim_energy_timing_models():
    """Test SRAM CIM energy and timing calculations"""
    from ramwich.config.data_config import DataConfig, BitConfig
    
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM],
        weight_format="Q0.2",  # 2 SRAM units = 2 bits total
        activation_format="Q8.8"
    )
    
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=data_config
    )
    
    mvmu = MVMU(id=0, config=config)
    
    # Execute some operations
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    weights = np.random.randint(-1, 2, size=(xbar_size, xbar_size)).astype(np.float64)
    mvmu.load_weights(weights)
    
    input_data = np.random.randint(0, 8, size=xbar_size).astype(np.int32)
    mvmu.write_to_inreg(0, input_data)
    
    mvmu.execute_mvm()
    
    # Check energy and timing stats
    stats = mvmu.get_stats()
    sram_stats = stats["SRAM CIM Unit"]
    
    # Verify energy calculations
    assert sram_stats.dynamic_energy > 0
    assert sram_stats.leakage_energy > 0
    assert sram_stats.area > 0
    assert sram_stats.activation_count > 0
    
    print(f"SRAM CIM Dynamic Energy: {sram_stats.dynamic_energy}")
    print(f"SRAM CIM Leakage Energy: {sram_stats.leakage_energy}")
    print(f"SRAM CIM Area: {sram_stats.area}")
    print(f"SRAM CIM Activations: {sram_stats.activation_count}")


def test_sram_cim_accuracy():
    """Test SRAM CIM computational accuracy"""
    from ramwich.config.data_config import DataConfig, BitConfig
    
    data_config = DataConfig(
        storage_config=[BitConfig.SRAM],
        weight_format="Q1.0",  # 1 SRAM unit = 1 bit total
        activation_format="Q8.0"  # No fractional bits to match weight format
    )
    
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=data_config
    )
    
    mvmu = MVMU(id=0, config=config)
    
    # Use simple test case for accuracy verification
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    
    # Create simple weights matrix (identity-like for easy verification)
    weights = np.zeros((xbar_size, xbar_size), dtype=np.float64)
    np.fill_diagonal(weights, 1.0)  # Identity matrix
    
    mvmu.load_weights(weights)
    
    # Simple input vector
    input_data = np.arange(1, xbar_size + 1, dtype=np.int32)
    mvmu.write_to_inreg(0, input_data)
    
    mvmu.execute_mvm()
    output = mvmu.read_from_outreg(0, xbar_size)
    
    # For identity matrix multiplication, output should be close to input
    # (accounting for quantization and bit-width differences)
    expected = input_data
    
    # Allow some tolerance due to fixed-point arithmetic
    tolerance = max(1, np.max(expected) * 0.1)
    
    print(f"Input: {input_data[:10]}...")
    print(f"Output: {output[:10]}...")
    print(f"Expected: {expected[:10]}...")
    
    # Check that most values are reasonably close
    close_values = np.abs(output - expected) <= tolerance
    accuracy = np.mean(close_values)
    
    print(f"Accuracy: {accuracy * 100:.1f}%")
    assert accuracy > 0.8, f"SRAM CIM accuracy too low: {accuracy * 100:.1f}%"


if __name__ == "__main__":
    test_sram_cim_basic_functionality()
    test_sram_cim_mixed_configuration()
    test_sram_cim_energy_timing_models()
    test_sram_cim_accuracy()
    print("All SRAM CIM tests passed!")