import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import MVMUConfig, XBARConfig
from ..stats import Stats, StatsDict


class SRAMCIMUnitStats(BaseModel):
    # Universal metrics
    config: XBARConfig = Field(default=XBARConfig(), description="Xbar configuration")
    num_xbar: int = Field(default=0, description="Number of crossbars")
    num_calculator_per_xbar: int = Field(default=0, description="Number of calculators")

    # Xbar specific metrics
    mac_operations: int = Field(default=0, description="Total number of operations")

    def reset(self):
        """Reset all statistics to zero"""
        self.mac_operations = 0

    def get_stats(self) -> StatsDict:
        # Map Xbar metrics to Stat object
        stats = Stats(
            activation_count=self.mac_operations,
            dynamic_energy=(self.config.sram_xbar_pow_dyn + self.config.calculator_pow_dyn) * self.mac_operations,
            leakage_energy=(
                self.config.sram_xbar_pow_leak + self.config.calculator_pow_leak * self.num_calculator_per_xbar
            )
            * self.num_xbar,
            area=(self.config.sram_xbar_area + self.num_calculator_per_xbar) * self.num_xbar,
        )

        return StatsDict({"SRAM CIM Unit": stats})


class SRAMCIMUnitArray:
    """
    Crossbar array component that performs matrix-vector multiplication operations.
    """

    def __init__(self, mvmu_config: MVMUConfig):
        self.mvmu_config = mvmu_config
        self.xbar_config = self.mvmu_config.xbar_config
        self.num_xbar = self.mvmu_config.num_sram_xbar_per_mvmu
        self.xbar_size = self.xbar_config.xbar_size
        self.num_calculator_per_xbar = self.xbar_size // self.mvmu_config.num_columns_per_calculator

        # Initialize the crossbar
        self.pos_xbar = np.zeros((self.num_xbar, self.xbar_size, self.xbar_size)).astype(np.int8)
        self.neg_xbar = np.zeros((self.num_xbar, self.xbar_size, self.xbar_size)).astype(np.int8)

        # Initialize stats
        self.stats = SRAMCIMUnitStats(
            config=self.xbar_config, num_xbar=self.num_xbar * 2, num_calculator_per_xbar=self.num_calculator_per_xbar
        )  # 2 for pos and neg xbar

    def load_weights(self, weights: NDArray[np.int32]):
        """Load ternary weights (0, -1, 1) into the crossbar"""

        expected_shape = (self.num_xbar, self.xbar_size, self.xbar_size)
        if weights.shape != expected_shape:
            raise ValueError(f"Expected weights shape {expected_shape}, got {weights.shape}")

        # Direct boolean comparisons are faster than np.maximum
        self.pos_xbar = (weights == 1).astype(np.int8)
        self.neg_xbar = (weights == -1).astype(np.int8)

    def execute_mvm(self, input_vector: NDArray[np.int32]):
        """Execute a matrix-vector multiplication operation

        Args:
            input_vector: 1D array of length xbar_size representing the input values

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
        pos_result = np.einsum("ikj,j->ik", self.pos_xbar, input_vector)
        neg_result = np.einsum("ikj,j->ik", self.neg_xbar, input_vector)

        result = pos_result - neg_result

        # Update the statistics
        self.stats.mac_operations += (
            self.num_xbar * 2 * self.xbar_size
        )  # Two operations per crossbar (one for pos and one for neg)

        return result

    def reset(self):
        """Reset all statistics to zero"""
        self.stats.reset()

    def get_stats(self) -> StatsDict:
        """Get statistics for this Xbar"""
        return self.stats.get_stats()
