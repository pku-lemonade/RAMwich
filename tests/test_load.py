"""
Test module for data loading functionality in RAMwich.
"""
import pytest
from ramwich import RAMwich
from ramwich.ops import Send


@pytest.mark.unit
def test_operation_loading(ramwich_simulator):
    """Test that operations are loaded correctly from JSON file."""
    simulator = ramwich_simulator
    
    # Validate that the first operation is loaded correctly
    validate_op = Send(type="send", node=0, tile=0, mem_addr=768, target_tile=3, width=16, vec=1)
    actual_op = simulator.get_node(0).get_tile(0).operations[0]
    
    assert actual_op == validate_op, "Operation not loaded correctly"


@pytest.mark.unit
def test_weight_loading(ramwich_simulator):
    """Test that weights are loaded correctly from NPZ file."""
    simulator = ramwich_simulator
    
    # Validate that weights are loaded correctly
    weight_value = simulator.get_node(0).get_tile(2).cores[0].mvmus[0].rram_xbar_array.neg_xbar[0][0][0]
    assert weight_value == 1, "Weight not loaded correctly"


@pytest.mark.integration
def test_complete_loading_pipeline(test_config_file, test_ops_file, test_weights_file):
    """Test the complete loading pipeline with all components."""
    simulator = RAMwich(
        config_file=test_config_file,
        ops_file=test_ops_file,
        weights_file=test_weights_file,
        quiet=True
    )
    
    # Verify simulator is properly initialized
    assert simulator is not None
    assert len(simulator.nodes) > 0
    assert simulator.nodes[0].tiles is not None
    
    # Verify operations are loaded
    tile = simulator.get_node(0).get_tile(0)
    assert len(tile.operations) > 0
    
    # Verify weights are loaded
    core = simulator.get_node(0).get_tile(2).get_core(0)
    mvmu = core.get_mvmu(0)
    assert mvmu.rram_xbar_array is not None


if __name__ == "__main__":
    # Backward compatibility for direct script execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    config_file = "examples/mlp_l4_mnist/config.yaml"
    ops_file = "examples/mlp_l4_mnist/ops.json"
    weights_file = "examples/mlp_l4_mnist/weights.npz"

    simulator = RAMwich(config_file=config_file, ops_file=ops_file, weights_file=weights_file)

    validate_op = Send(type="send", node=0, tile=0, mem_addr=768, target_tile=3, width=16, vec=1)
    assert simulator.get_node(0).get_tile(0).operations[0] == validate_op, "Operation not loaded correctly"
    assert (
        simulator.get_node(0).get_tile(2).cores[0].mvmus[0].rram_xbar_array.neg_xbar[0][0][0] == 1
    ), "Weight not loaded correctly"
    print("Operation and weight loaded correctly.")
