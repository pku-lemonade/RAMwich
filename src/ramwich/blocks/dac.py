from typing import Any, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import MVMUConfig
from ..stats import Stats


class DACStats(BaseModel):
    """Statistics tracking for DAC (Digital-to-Analog Converter) components"""

    # Basic metrics
    conversions: int = Field(default=0, description="Number of D/A conversions performed")

    # Performance metrics
    active_cycles: int = Field(default=0, description="Number of active cycles")
    energy_consumption: float = Field(default=0.0, description="Energy consumption in pJ")

    def get_stats(self) -> Stats:
        """Get DAC-specific statistics"""
        stats = Stats()

        # Map DAC metrics to Stat object
        stats.latency = float(self.active_cycles)
        stats.energy = float(self.energy_consumption)
        stats.area = 0.0  # Set appropriate area value if available

        # Map operation counts
        stats.operations = self.conversions

        # Set execution time metrics
        stats.total_execution_time = float(self.active_cycles)

        return stats


class DACArray:
    """Hardware implementation of the DAC component"""

    def __init__(self, mvmu_config: MVMUConfig = None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.dac_config = self.mvmu_config.dac_config
        self.size = self.mvmu_config.xbar_config.xbar_size

        # Calculate max value based on resolution
        self.max_value = (1 << self.mvmu_config.dac_config.resolution) - 1

        # Initialize stats
        self.stats = DACStats()

    def convert(self, digital_value: NDArray[np.int32]):
        """Simulate DAC conversion from digital to analog"""

        # Validate input
        if digital_value.ndim != 1:
            raise ValueError(f"Expected 1D array, got {digital_value.ndim}D array")

        if len(digital_value) != self.size:
            raise ValueError(f"Expected input vector of shape ({self.size},), got {digital_value.shape}")

        # Apply analog conversion based on resolution
        fraction = digital_value / self.max_value
        clipped_value = np.clip(fraction, 0, 1)
        analog_value = clipped_value * self.mvmu_config.dac_config.VDD

        # Update stats
        self.stats.conversions += self.size
        self.stats.active_cycles += self.dac_config.lat
        self.stats.energy_consumption += self.dac_config.pow_dyn * self.size

        return analog_value

    def get_energy_consumption(self):
        """Return the total energy consumption in pJ"""
        return self.stats.energy_consumption

    def get_stats(self) -> Stats:
        """Return detailed statistics about this DAC"""
        return self.stats.get_stats(self.dac_id)
