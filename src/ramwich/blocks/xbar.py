from typing import List

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import MVMUConfig, XBARConfig, DataConfig
from ..stats import Stats


class XbarStats(BaseModel):
    operations: int = Field(default=0, description="Total number of operations")
    mvm_operations: int = Field(default=0, description="Number of MVM operations")

    def get_stats(self) -> Stats:
        stats = Stats()
        stats.latency = 0.0
        stats.energy = 0.0
        stats.area = 0.0
        stats.operations = self.operations
        stats.mvm_operations = self.mvm_operations
        return stats


class XbarArray:
    """
    Crossbar array component that performs matrix-vector multiplication operations.
    """

    def __init__(self, mvmu_config: MVMUConfig=None, data_config: DataConfig=None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.data_config = data_config or DataConfig()
        self.xbar_config = self.mvmu_config.xbar_config
        self.num_xbar = data_config.num_rram_xbar_per_matrix
        self.xbar_size = self.xbar_config.xbar_size

        # Initialize the crossbar
        self.pos_xbar = np.zeros((self.num_xbar, self.xbar_size, self.xbar_size))
        self.neg_xbar = np.zeros((self.num_xbar, self.xbar_size, self.xbar_size))

        # Initialize stats
        self.stats = XbarStats()

    def load_weights(self, weights: NDArray[np.floating]):
        """Load weights into the crossbar"""

        expected_shape = (self.num_xbar, self.xbar_size, self.xbar_size)
        if weights.shape != expected_shape:
            raise ValueError(f"Expected weights shape {expected_shape}, got {weights.shape}")
        
        self.pos_xbar = np.maximum(weights, 0)
        self.neg_xbar = np.maximum(-weights, 0)

    def __repr__(self):
        return f"Xbar({self.id}, size={self.size})"

    def execute_mvm(self, xbar_data):
        """Execute a matrix-vector multiplication operation"""
        self.stats.operations += 1
        self.stats.mvm_operations += 1
        return True

    def update_execution_time(self, execution_time):
        """Update the execution time statistics"""
        self.stats.latency += execution_time

    def get_stats(self) -> Stats:
        """Get statistics for this Xbar"""
        return self.stats.get_stats()

    def set_values(self, values: List[int]):
        """Set values in the crossbar"""
        for i, val in enumerate(values):
            if i < self.size:
                self.memory[i] = val
            else:
                break
