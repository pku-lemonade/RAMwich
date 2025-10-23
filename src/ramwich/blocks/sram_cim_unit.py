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
    exp_operations: int = Field(default=0, description="Total number of exponential operations")

    def reset(self):
        """Reset all statistics to zero"""
        self.mac_operations = 0
        self.exp_operations = 0

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

        # Use precomputed types and indices from config to avoid per-unit recomputation
        self.xbar_types = list(self.mvmu_config.sram_xbar_types)
        self.mvm_indices = np.array(self.mvmu_config.sram_mvm_indices, dtype=int)
        # EXP-style computation is required for both EXP and MANT at the last iteration
        self.exp_indices = np.array(self.mvmu_config.sram_exp_indices, dtype=int)

        # Initialize exp input buffer for exponential calculations if needed
        if len(self.exp_indices) > 0:
            self.exp_input = np.zeros(self.xbar_size, dtype=np.int64)

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

    def execute(self, input_vector: NDArray[np.int32], iteration: int):
        """Execute a matrix-vector multiplication operation

        Args:
            input_vector: 1D array of length xbar_size representing the input values
            iteration: Current iteration number (0-indexed)

        Returns:
            Tuple of two 2D arrays, each with shape (num_xbar, xbar_size):
            - mvm_result: MVM outputs (needs iteration shift in SNA)
            - exp_result: EXP outputs (no iteration shift in SNA)
        """

        # Validate input
        if input_vector.ndim != 1:
            raise ValueError(f"Expected 1D array, got {input_vector.ndim}D array")

        if len(input_vector) != self.xbar_size:
            raise ValueError(f"Expected input vector of shape ({self.xbar_size},), got {input_vector.shape}")

        # Update current iteration
        is_last_iteration = iteration == self.mvmu_config.num_iterations - 1

        # Initialize separate result arrays for MVM and EXP
        mvm_result = np.zeros((self.num_xbar, self.xbar_size)).astype(np.int64)
        exp_result = np.zeros((self.num_xbar, self.xbar_size)).astype(np.int64)

        # Perform MVM only for xbars that need it (INT and MANT types)
        if len(self.mvm_indices) > 0:
            # Use einsum for efficient matrix-vector multiplication on masked xbars
            # i: crossbar index, j: crossbar row, k: crossbar column (multiplied by input)
            pos_result = np.einsum("ikj,j->ik", self.pos_xbar[self.mvm_indices], input_vector)
            neg_result = np.einsum("ikj,j->ik", self.neg_xbar[self.mvm_indices], input_vector)
            mvm_output = pos_result - neg_result

            # Process INT and MANT xbars - store in mvm_result
            for i, xbar_idx in enumerate(self.mvm_indices):
                mvm_result[xbar_idx] += mvm_output[i]

        # Process EXP xbars separately (only shift-and-add, no MVM)
        if len(self.exp_indices) > 0:
            # For EXP, we do not need MVM, but need to store input shifted by iteration
            self.exp_input += input_vector << (iteration * self.mvmu_config.dac_config.resolution)

            # For last iteration, compute the final exponential result
            if is_last_iteration:
                # Compute per-row sum over columns of (weight << input)
                # Broadcast shift across the column dimension, then reduce along columns
                pos_shifted = self.pos_xbar[self.exp_indices].astype(np.int64) << self.exp_input
                neg_shifted = self.neg_xbar[self.exp_indices].astype(np.int64) << self.exp_input
                pos_sum = np.sum(pos_shifted, axis=2)
                neg_sum = np.sum(neg_shifted, axis=2)
                exp_output = pos_sum - neg_sum
                for i, xbar_idx in enumerate(self.exp_indices):
                    # Return accumulated exponential result on last iteration in exp_result
                    exp_result[xbar_idx] += exp_output[i]
                    # Reset input buffer for next forward pass
                self.exp_input.fill(0)
            # else: exp_result remains 0 for non-last iterations, no action needed

        # Update the statistics (only count actual operations performed)
        num_mvm_xbars = len(self.mvm_indices)
        self.stats.mac_operations += num_mvm_xbars * 2 * self.xbar_size
        if is_last_iteration:
            num_exp_xbars = len(self.exp_indices)
            self.stats.exp_operations += num_exp_xbars * 2 * self.xbar_size

        return mvm_result, exp_result

    def reset(self):
        """Reset all statistics and accumulators to zero"""
        self.stats.reset()
        if hasattr(self, "exp_input"):
            self.exp_input.fill(0)

    def get_stats(self) -> StatsDict:
        """Get statistics for this Xbar"""
        return self.stats.get_stats()
