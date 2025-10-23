import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import MVMUConfig
from ..stats import Stats, StatsDict


class SNAStats(BaseModel):
    """Statistics tracking for SNA (Shift and Add) components"""

    # Config
    config: MVMUConfig = Field(default=MVMUConfig(), description="Configuration object")

    # SNA specific metrics
    operations: int = Field(default=0, description="Number of operations performed")
    active_cycles: int = Field(default=0, description="Number of active cycles")

    def reset(self):
        """Reset all statistics to zero"""
        self.operations = 0
        self.active_cycles = 0

    def get_stats(self) -> StatsDict:
        """Convert SNAStats to general Stats object"""
        stats = Stats(
            activation_count=self.operations,
            dynamic_energy=self.config.sna_pow_dyn * self.operations,
            leakage_energy=self.config.sna_pow_leak,
            area=self.config.sna_area,
        )

        return StatsDict({"Shift and Add": stats})


class SNAArray:
    """Hardware implementation of Shift and Add (SNA) array"""

    def __init__(self, mvmu_config: MVMUConfig):
        self.mvmu_config = mvmu_config
        self.input_shape = (self.mvmu_config.num_xbar_per_mvmu, self.mvmu_config.num_adc_per_xbar)
        self.size = np.prod(self.input_shape) + self.mvmu_config.num_adc_per_xbar

        # Precompute per-xbar shift relative to its weight partition.
        # Previously we used absolute stored_bit offsets, which incorrectly
        # introduced inter-partition shifts when weights are partitioned.
        # To make partitions operate in parallel, subtract the partition base.
        stored_bit = np.array(self.mvmu_config.stored_bit[:-1], dtype=int)
        wp = np.array(self.mvmu_config.data_config.weight_partition, dtype=int)
        # For each xbar k, find the partition index such that wp[idx] <= stored_bit[k]
        # Using searchsorted to get the rightmost insertion position, then minus 1
        part_idx = np.searchsorted(wp, stored_bit, side="right") - 1
        part_bases = wp[part_idx]
        # Shift within partition only (no cross-partition offset)
        self.shift_bits = (stored_bit - part_bases)[:, np.newaxis]

        # Initialize stats
        self.stats = SNAStats(config=self.mvmu_config)

    def calculate(self, input_data: NDArray[np.int32], current_value: NDArray[np.int32], bits: int):
        """Performs the Shift and Add (SNA) operation on the input data

        Each SNA unit combines data from multiple xbars for one ADC position:
        - Takes the values from all xbars for that specific ADC
        - Applies appropriate bit shifts based on xbar position
        - Adds the shifted values to produce a single output

        Args:
            input_data: 2D array with shape (num_xbar_per_mvmu, num_adc_per_xbar)
            bits_per_cell: Array of bit shifts to apply for each xbar (optional)

        Returns:
            1D array with shape (num_adc_per_xbar,) containing SNA results
        """
        # Validate input data
        if input_data.shape != self.input_shape:
            raise ValueError(f"Input data shape {input_data.shape} does not match SNA array shape {self.input_shape}")

        if len(current_value) != self.mvmu_config.num_adc_per_xbar:
            raise ValueError(
                f"Current value length {len(current_value)} does not match SNA array shape {self.mvmu_config.num_adc_per_xbar}"
            )

        if bits < 0:
            raise ValueError(f"Bits {bits} must be non-negative")
        # Apply intra-partition shifts for each xbar (parallel partitions)
        shifted_data = input_data.astype(np.int64) << self.shift_bits  # Prevent overflow during shifting

        # Sum across the xbar dimension to get final results
        result = np.sum(shifted_data, axis=0)

        # Shift the accumulated result by the activation bit-slice position
        # (this is independent of weight partitions)
        result = result << bits

        # Update stats
        self.stats.operations += self.size

        return result + current_value

    def reset(self):
        """Reset all statistics to zero"""
        self.stats.reset()

    def get_stats(self) -> StatsDict:
        """Convert SNAStats to general Stats object"""
        return self.stats.get_stats()
