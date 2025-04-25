from typing import List

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import DataConfig, MVMUConfig, XBARConfig
from ..stats import Stats


class XbarStats(BaseModel):

    # Universal metrics
    unit_energy_consumption: float = Field(default=0.0, description="Energy consumption for each convertion in pJ")
    leakage_energy_per_cycle: float = Field(default=0.0, description="Leakage energy consumption for 1 cycle in pJ")
    area: float = Field(default=0.0, description="Area in mm^2")

    # Xbar specific metrics
    mvm_operations: int = Field(default=0, description="Total number of operations")

    def get_stats(self) -> Stats:
        stats = Stats()

        # Map Xbar metrics to Stat object
        stats.dynamic_energy = self.unit_energy_consumption * self.mvm_operations
        stats.leakage_energy = self.leakage_energy_per_cycle
        stats.area = self.area

        stats.increment_component_count("Xbar", self.mvm_operations)

        return stats


class XbarArray:
    """
    Crossbar array component that performs matrix-vector multiplication operations.
    """

    def __init__(self, mvmu_config: MVMUConfig = None):
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
        self.stats.unit_energy_consumption = self.xbar_config.xbar_pow
        self.stats.leakage_energy_per_cycle = self.xbar_config.xbar_pow_leak * self.num_xbar * 2
        self.stats.area = self.xbar_config.xbar_area * self.num_xbar * 2  # 2 for pos and neg xbar

    def load_weights(self, weights: NDArray[np.float64]):
        """Load weights into the crossbar"""

        expected_shape = (self.num_xbar, self.xbar_size, self.xbar_size)
        if weights.shape != expected_shape:
            raise ValueError(f"Expected weights shape {expected_shape}, got {weights.shape}")

        self.pos_xbar = np.maximum(weights, 0)
        self.neg_xbar = np.maximum(-weights, 0)

    def __repr__(self):
        return f"Xbar({self.id}, size={self.size})"

    def execute_mvm(self, input_vector: NDArray[np.float64]):
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

        if len(input_vector) != self.xbar_size:
            raise ValueError(f"Expected input vector of shape ({self.xbar_size},), got {input_vector.shape}")

        # Use einsum for efficient matrix-vector multiplication across all crossbars
        # i: crossbar index, j: crossbar row, k: crossbar column (multiplied by input)
        # 'ijk,k->ij' for transpose multiplication
        # 'ikj,j->ik' for standard multiplication
        if self.has_noise:
            rng = np.random.default_rng()
            pos_mat = self.pos_xbar + rng.normal(0, self.xbar_config.noise_sigma, self.pos_xbar.shape)
            neg_mat = self.neg_xbar + rng.normal(0, self.xbar_config.noise_sigma, self.neg_xbar.shape)
            pos_result = np.einsum("ikj,j->ik", pos_mat, input_vector)
            neg_result = np.einsum("ikj,j->ik", neg_mat, input_vector)
        else:
            pos_result = np.einsum("ikj,j->ik", self.pos_xbar, input_vector)
            neg_result = np.einsum("ikj,j->ik", self.neg_xbar, input_vector)

        # Update the statistics
        self.stats.mvm_operations += self.num_xbar * 2  # Two operations per crossbar (one for pos and one for neg)

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
