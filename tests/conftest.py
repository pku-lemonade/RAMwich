"""
Pytest configuration and shared fixtures for RAMwich tests.
"""
import os
import sys
from pathlib import Path

import pytest
import numpy as np

# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set up test data directory
TEST_DATA_DIR = project_root / "examples" / "mlp_l4_mnist"


@pytest.fixture(scope="session")
def test_config_file():
    """Provide path to test configuration file."""
    config_path = TEST_DATA_DIR / "config.yaml"
    if not config_path.exists():
        pytest.skip(f"Test configuration file not found: {config_path}")
    return str(config_path)


@pytest.fixture(scope="session")
def test_ops_file():
    """Provide path to test operations file."""
    ops_path = TEST_DATA_DIR / "ops.json"
    if not ops_path.exists():
        pytest.skip(f"Test operations file not found: {ops_path}")
    return str(ops_path)


@pytest.fixture(scope="session")
def test_weights_file():
    """Provide path to test weights file."""
    weights_path = TEST_DATA_DIR / "weights.npz"
    if not weights_path.exists():
        pytest.skip(f"Test weights file not found: {weights_path}")
    return str(weights_path)


@pytest.fixture(scope="session")
def test_activation_file():
    """Provide path to test activation file."""
    activation_path = TEST_DATA_DIR / "activation.npy"
    if not activation_path.exists():
        pytest.skip(f"Test activation file not found: {activation_path}")
    return str(activation_path)


@pytest.fixture
def sample_input_vector():
    """Provide a sample input vector for testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.randint(0, 2**15 - 1, size=(128,), dtype=np.int32)


@pytest.fixture
def ramwich_simulator(test_config_file, test_ops_file, test_weights_file):
    """Provide a configured RAMwich simulator instance."""
    from ramwich import RAMwich
    
    return RAMwich(
        config_file=test_config_file,
        ops_file=test_ops_file,
        weights_file=test_weights_file,
        quiet=True  # Suppress output during tests
    )


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return str(output_dir)


# Performance testing utilities
@pytest.fixture
def performance_threshold():
    """Define performance thresholds for tests."""
    return {
        "simulation_time": 30.0,  # seconds
        "memory_usage": 1024 * 1024 * 1024,  # 1GB
        "accuracy_threshold": 0.5,  # 50% error tolerance for hardware simulation
    }


# Skip markers for optional dependencies
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "visualization: mark test as requiring visualization libraries"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark visualization tests
        if "visualization" in item.nodeid:
            item.add_marker(pytest.mark.visualization)
        
        # Mark integration tests
        if any(keyword in item.nodeid for keyword in ["integration", "end_to_end", "full"]):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if any(keyword in item.nodeid for keyword in ["mnist", "resnet", "large"]):
            item.add_marker(pytest.mark.slow)