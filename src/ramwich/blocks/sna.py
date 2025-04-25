import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import MVMUConfig
from ..stats import Stats


class SNAStats(BaseModel):
    """Statistics tracking for SNA (Shift and Add) components"""

    # Universal metrics
    unit_energy_consumption: float = Field(default=0.0, description="Energy consumption for each operation in pJ")
    leakage_energy_per_cycle: float = Field(default=0.0, description="Leakage energy consumption for 1 cycle in pJ")
    area: float = Field(default=0.0, description="Area in mm^2")

    # SNA specific metrics
    operations: int = Field(default=0, description="Number of operations performed")
    active_cycles: int = Field(default=0, description="Number of active cycles")

    def get_stats(self) -> Stats:
        """Convert SNAStats to general Stats object"""
        stats = Stats()

        # Map SNA metrics to Stat object
        stats.dynamic_energy = self.unit_energy_consumption * self.operations
        stats.leakage_energy = self.leakage_energy_per_cycle
        stats.area = self.area

        stats.increment_component_count("SNA", self.operations)

        return stats


class SNAArray:
    """Hardware implementation of Shift and Add (SNA) array"""

    def __init__(self, mvmu_config: MVMUConfig = None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.input_shape = (self.mvmu_config.num_xbar_per_mvmu, self.mvmu_config.num_adc_per_xbar)
        self.size = np.prod(self.input_shape) + self.mvmu_config.num_adc_per_xbar

        self.shift_bits = np.array(self.mvmu_config.stored_bit[:-1])[:, np.newaxis]

        # Initialize stats
        self.stats = SNAStats()
        self.unit_energy_consumption = self.mvmu_config.sna_pow_dyn
        self.leakage_energy_per_cycle = self.mvmu_config.sna_pow_leak
        self.area = self.mvmu_config.sna_area * self.mvmu_config.num_rram_xbar_per_mvmu

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
            raise ValueError(f"Input data shape {input_data.shape} does not match SNA array shape {self.shape}")

        if len(current_value) != self.mvmu_config.num_adc_per_xbar:
            raise ValueError(
                f"Current value length {len(current_value)} does not match SNA array shape {self.mvmu_config.num_adc_per_xbar}"
            )

        if bits < 0:
            raise ValueError(f"Bits {bits} must be non-negative")

        # Apply shifts for each xbar
        shifted_data = input_data.astype(np.int64) << self.shift_bits  # Prevent overflow during shifting

        # Sum across the xbar dimension to get final results
        result = np.sum(shifted_data, axis=0)

        # shift the result to the left by bits
        result = result << bits

        # Update stats
        self.stats.operations += self.size

        return result + current_value

    def get_stats(self) -> Stats:
        """Convert SNAStats to general Stats object"""
        return self.stats.get_stats()
