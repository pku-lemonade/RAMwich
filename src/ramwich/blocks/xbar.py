from typing import List

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import MVMUConfig, XBARConfig, DataConfig
from ..stats import Stats


class XbarStats(BaseModel):
    operations: int = Field(default=0, description="Total number of operations")

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

    def __init__(self, mvmu_config: MVMUConfig=None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.xbar_config = self.mvmu_config.xbar_config
        self.num_xbar = self.mvmu_config.num_rram_xbar_per_mvmu
        self.xbar_size = self.xbar_config.xbar_size
        self.has_noise = self.xbar_config.has_noise

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

    def execute_mvm(self, input_vector: NDArray[np.floating]):
        """Execute a matrix-vector multiplication operation

        Args:
            input_vector: 1D array of length xbar_size representing the input voltages
        
        Returns:
            2D array with shape (num_xbar, xbar_size) containing the results of matrix-vector 
            multiplication for each crossbar
        """

        # Validate input
        if input_vector.ndim != 1:
            raise ValueError(f"Expected 1D array, got {input_vector.ndim}D array")

        if len(input_vector) != self.size:
            raise ValueError(f"Expected input vector of shape ({self.xbar_size},), got {input_vector.shape}")
        
        # Use einsum for efficient matrix-vector multiplication across all crossbars
        # 'ijk,k->ij' means: sum the product over the last dimension (k)
        # i: crossbar index, j: crossbar row, k: crossbar column (multiplied by input)
        if self.has_noise:
            rng = np.random.default_rng()
            pos_mat = self.pos_xbar + rng.normal(0, self.xbar_config.noise_sigma, self.pos_xbar.shape)
            neg_mat = self.neg_xbar + rng.normal(0, self.xbar_config.noise_sigma, self.neg_xbar.shape)          
            pos_result = np.einsum('ijk,k->ij', pos_mat, input_vector)
            neg_result = np.einsum('ijk,k->ij', neg_mat, input_vector)
        else:
            pos_result = np.einsum('ijk,k->ij', self.pos_xbar, input_vector)
            neg_result = np.einsum('ijk,k->ij', self.neg_xbar, input_vector)

        # Update the statistics
        self.stats.operations += self.num_xbar * 2  # Two operations per crossbar (one for pos and one for neg)

        return pos_result, neg_result

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
