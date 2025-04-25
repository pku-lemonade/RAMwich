import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import MVMUConfig
from ..stats import Stats


class MUXStats(BaseModel):
    """Statistics tracking for ADC (Analog-to-Digital Converter) components"""

    # Universal metrics
    unit_energy_consumption: float = Field(default=0.0, description="Energy consumption for each convertion in pJ")
    leakage_energy_per_cycle: float = Field(default=0.0, description="Leakage energy consumption for 1 cycle in pJ")
    area: float = Field(default=0.0, description="Area in mm^2")

    # MUX specific metrics
    selections: int = Field(default=0, description="Number of selections performed")

    def get_stats(self) -> Stats:
        """Convert ADCStats to general Stats object"""
        stats = Stats()

        # Map ADC metrics to Stat object
        stats.increment_component_activation("MUX", self.selections)
        stats.increment_component_dynamic_energy("MUX", self.unit_energy_consumption * self.selections)
        stats.increment_component_leakage_energy("MUX", self.leakage_energy_per_cycle)
        stats.increment_component_area("MUX", self.area)

        return stats


class MuxArray:
    """Hardware implementation of a multiplexer array for selecting specific elements"""

    def __init__(self, mvmu_config: MVMUConfig = None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.num_xbar = self.mvmu_config.num_rram_xbar_per_mvmu
        self.num_input_per_mux = self.mvmu_config.num_columns_per_adc
        self.num_mux_per_xbar = self.mvmu_config.num_adc_per_xbar

        # Define 2D shape

        self.input_shape = (self.num_xbar, self.mvmu_config.xbar_config.xbar_size)
        self.output_shape = (self.num_xbar, self.num_mux_per_xbar)
        self.size = np.prod(self.output_shape)

        # Initialize stats
        self.stats = MUXStats()
        self.stats.unit_energy_consumption = self.mvmu_config.mux_pow_dyn
        self.stats.leakage_energy_per_cycle = self.mvmu_config.mux_pow_leak
        self.stats.area = self.mvmu_config.mux_area * self.size

    def select(self, input_array: NDArray[np.float64], index: int):
        """Selects the value at the given index from the input array using a multiplexer"""

        # Validate input
        if input_array.shape != self.input_shape:
            raise ValueError(f"Expected input array of shape {self.input_shape}, got {input_array.shape}")

        if index < 0 or index >= self.num_input_per_mux:
            raise ValueError(f"Index {index} out of bounds")

        reshaped_input = input_array.reshape(self.num_xbar, self.num_mux_per_xbar, self.num_input_per_mux)

        # Update stats
        self.stats.selections += self.size

        return reshaped_input[:, :, index]

    def get_stats(self) -> Stats:
        """Get the statistics for this MUX array"""
        return self.stats.get_stats()
