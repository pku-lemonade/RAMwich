import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import MVMUConfig, XBARConfig
from ..stats import Stats, StatsDict


class SRAMCIMUnitStats(BaseModel):
    # Universal metrics
    config: XBARConfig = Field(default=XBARConfig(), description="Xbar configuration")
    num_xbar: int = Field(default=0, description="Number of crossbars")
    num_smacu: int = Field(default=0, description="Number of SRAM CIM shifter-mac units")
    num_macu_per_xbar: int = Field(default=0, description="Number of MAC units per crossbar")

    # Xbar specific metrics
    mac_operations: int = Field(default=0, description="Total number of MAC operations")
    eaa_operations: int = Field(default=0, description="Total number of exponential operations")
    ewmvm_operations: int = Field(default=0, description="Total number of exponential with MVM operations")

    def reset(self):
        """Reset all statistics to zero"""
        self.mac_operations = 0
        self.eaa_operations = 0
        self.ewmvm_operations = 0

    def get_stats(self) -> StatsDict:
        # Map Xbar metrics to Stat object
        # Dynamic energy:
        # - mac_operations: linear MVM on INT/MANT -> xbar + calculator + mac
        # - eaa_operations: EXP(A) shift-and-accumulate on last iter -> xbar + calculator (no MAC)
        # - ewmvm_operations: EXP(W) MVM on all iters -> xbar + calculator + mac
        dyn_xbar = self.config.sram_xbar_pow_dyn * (self.mac_operations + self.eaa_operations + self.ewmvm_operations)
        dyn_mvm = self.config.mac_pow_dyn * self.mac_operations
        # dyn_eaa   = (self.config.sram_xbar_pow_dyn + self.config.calculator_pow_dyn * self.num_calculator_per_xbar) * self.eaa_operations
        dyn_ewmvm = self.config.smac_pow_dyn * self.ewmvm_operations
        stats = Stats(
            activation_count=self.mac_operations + self.eaa_operations + self.ewmvm_operations,
            dynamic_energy=dyn_xbar + dyn_mvm + dyn_ewmvm,
            leakage_energy=(self.config.sram_xbar_pow_leak + self.config.macu_pow_leak * self.num_macu_per_xbar)
            * self.num_xbar
            + self.config.smacu_pow_leak * self.num_smacu,
            area=(self.config.sram_xbar_area + self.num_macu_per_xbar * self.config.mac_area) * self.num_xbar
            + self.config.smacu_area * self.num_smacu,
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
        self.num_macu_per_xbar = self.xbar_size // self.mvmu_config.num_columns_per_macu

        # Initialize the crossbar
        self.pos_xbar = np.zeros((self.num_xbar, self.xbar_size, self.xbar_size)).astype(np.int8)
        self.neg_xbar = np.zeros((self.num_xbar, self.xbar_size, self.xbar_size)).astype(np.int8)

        # Use precomputed types and indices from config to avoid per-unit recomputation
        self.xbar_types = list(self.mvmu_config.sram_xbar_types)
        # MVM is required for INT and MANT xbars
        self.mvm_indices = np.array(self.mvmu_config.sram_mvm_indices, dtype=int)
        # EAA computation is required for EXPA at the last iteration
        self.eaa_indices = np.array(self.mvmu_config.sram_eaa_indices, dtype=int)
        # EWMVM computation is required for EXPW and MANT at all iterations
        self.ewmvm_indices = np.array(self.mvmu_config.sram_ewmvm_indices, dtype=int)
        # MANT xbars need scaling factor applied to their MVM part
        self.mant_indices = np.array(self.mvmu_config.sram_mant_indices, dtype=int)
        self.mant_factor = self.mvmu_config.data_config.MANT_factor
        # Create a boolean mask for fast MANT lookup: True if mvm_indices[i] is a MANT xbar
        self.mvm_is_mant = np.isin(self.mvm_indices, self.mant_indices)

        # Initialize eaa input buffer for exponential calculations if needed
        if len(self.eaa_indices) > 0:
            self.eaa_input = np.zeros(self.xbar_size, dtype=np.int64)

        # Initialize ewmvm weight buffer for exponential with MVM calculations if needed
        # This is not part of real hardware, just for simulation purposes
        # In real hardware, weights are stored in pos_xbar and neg_xbar, and shifter will read them directly
        if len(self.ewmvm_indices) > 0:
            self.ewmvm_weight = np.zeros(
                (len(self.mvmu_config.expw_partition_indices), self.xbar_size, self.xbar_size), dtype=np.int64
            )

        # Initialize stats
        self.stats = SRAMCIMUnitStats(
            config=self.xbar_config,
            num_xbar=self.num_xbar * 2,
            num_macu_per_xbar=self.num_macu_per_xbar * 2,
            num_smacu=len(self.mvmu_config.expw_partition_indices) * 2,
        )  # 2 for pos and neg xbar

    def load_weights(self, weights: NDArray[np.int32]):
        """Load ternary weights (0, -1, 1) into the crossbar"""

        expected_shape = (self.num_xbar, self.xbar_size, self.xbar_size)
        if weights.shape != expected_shape:
            raise ValueError(f"Expected weights shape {expected_shape}, got {weights.shape}")

        # Direct boolean comparisons are faster than np.maximum
        self.pos_xbar = (weights == 1).astype(np.int8)
        self.neg_xbar = (weights == -1).astype(np.int8)

        # For EXPW xbars, store weights in ewmvm_weight buffer for simulation purposes
        if len(self.ewmvm_indices) > 0:
            for i, part_indices in enumerate(self.mvmu_config.expw_partition_indices):
                pos_weight = np.zeros((self.xbar_size, self.xbar_size), dtype=np.int64)
                neg_weight = np.zeros((self.xbar_size, self.xbar_size), dtype=np.int64)
                for xbar_idx in part_indices:
                    pos_weight += self.pos_xbar[xbar_idx]
                    neg_weight += self.neg_xbar[xbar_idx]
                    pos_weight = pos_weight << 1
                    neg_weight = neg_weight << 1
                self.ewmvm_weight[i] = pos_weight - neg_weight

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
        eaa_result = np.zeros((self.num_xbar, self.xbar_size)).astype(np.int64)
        ewmvm_result = np.zeros(self.xbar_size).astype(np.int64)

        # Perform MVM only for xbars that need it (INT and MANT types)
        if len(self.mvm_indices) > 0:
            # Use einsum for efficient matrix-vector multiplication on masked xbars
            # i: crossbar index, j: crossbar row, k: crossbar column (multiplied by input)
            pos_result = np.einsum("ikj,j->ik", self.pos_xbar[self.mvm_indices], input_vector)
            neg_result = np.einsum("ikj,j->ik", self.neg_xbar[self.mvm_indices], input_vector)
            mvm_output = pos_result - neg_result

            # Process INT and MANT xbars - store in mvm_result
            # For MANT xbars, apply the MANT_factor to the linear (MVM) part
            # Use vectorized operations for efficiency
            scaled_output = np.where(self.mvm_is_mant[:, np.newaxis], self.mant_factor * mvm_output, mvm_output)
            for i, xbar_idx in enumerate(self.mvm_indices):
                mvm_result[xbar_idx] += scaled_output[i]

        # Process EXPA xbars separately (only shift-and-add, no MVM)
        if len(self.eaa_indices) > 0:
            # For EXP, we do not need MVM, but need to store input shifted by iteration
            self.eaa_input += input_vector << (iteration * self.mvmu_config.dac_config.resolution)

            # For last iteration, compute the final exponential result
            if is_last_iteration:
                # Compute per-row sum over columns of (weight << input)
                # Broadcast shift across the column dimension, then reduce along columns
                pos_shifted = self.pos_xbar[self.eaa_indices].astype(np.int64) << self.eaa_input
                neg_shifted = self.neg_xbar[self.eaa_indices].astype(np.int64) << self.eaa_input
                pos_sum = np.sum(pos_shifted, axis=2)
                neg_sum = np.sum(neg_shifted, axis=2)
                exp_output = pos_sum - neg_sum
                for i, xbar_idx in enumerate(self.eaa_indices):
                    # Return accumulated exponential result on last iteration in eaa_result
                    eaa_result[xbar_idx] += exp_output[i]
                    # Reset input buffer for next forward pass
                self.eaa_input.fill(0)
            # else: eaa_result remains 0 for non-last iterations, no action needed

        # Process EWMVM xbars (exponential weight MVM) at all iterations
        if len(self.ewmvm_indices) > 0:
            # Combine all partition contributions in a single contraction
            ewmvm_result = np.einsum("ikj,j->k", self.ewmvm_weight, input_vector)

        # Update the statistics (only count actual operations performed)
        num_mvm_xbars = len(self.mvm_indices)
        self.stats.mac_operations += num_mvm_xbars * 2 * self.xbar_size
        num_ewmvms = len(self.mvmu_config.expw_partition_indices)
        self.stats.ewmvm_operations += num_ewmvms * 2 * self.xbar_size
        if is_last_iteration:
            num_exp_xbars = len(self.eaa_indices)
            self.stats.eaa_operations += num_exp_xbars * 2 * self.xbar_size

        return mvm_result, eaa_result, ewmvm_result

    def reset(self):
        """Reset all statistics and accumulators to zero"""
        self.stats.reset()
        if hasattr(self, "eaa_input"):
            self.eaa_input.fill(0)

    def get_stats(self) -> StatsDict:
        """Get statistics for this Xbar"""
        return self.stats.get_stats()
