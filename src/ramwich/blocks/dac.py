import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import DACConfig, MVMUConfig
from ..stats import Stats, StatsDict


class DACStats(BaseModel):
    """Statistics tracking for DAC (Digital-to-Analog Converter) components"""

    # Config
    config: DACConfig = Field(default=DACConfig(), description="DAC configuration")
    size: int = Field(default=0, description="Size of the DAC array")

    # DAC specific metrics
    conversions: int = Field(default=0, description="Number of D/A conversions performed")
    active_cycles: int = Field(default=0, description="Number of active cycles")

    def reset(self):
        """Reset all statistics to zero"""
        self.conversions = 0
        self.active_cycles = 0

    def get_stats(self) -> Stats:
        """Convert DACStats to general Stats object"""
        stats = Stats(
            activation_count=self.conversions,
            dynamic_energy=self.config.pow_dyn * self.conversions,
            leakage_energy=self.config.pow_leak * self.size,
            area=self.config.area * self.size,
        )

        return StatsDict({"DAC": stats})


class DACArray:
    """Hardware implementation of the DAC component"""

    def __init__(self, mvmu_config: MVMUConfig):
        self.mvmu_config = mvmu_config
        self.dac_config = self.mvmu_config.dac_config
        self.size = self.mvmu_config.xbar_config.xbar_size

        # Calculate max value based on resolution
        self.max_value = (1 << self.mvmu_config.dac_config.resolution) - 1

        # Initialize stats
        self.stats = DACStats(config=self.dac_config, size=self.size)

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

        return analog_value

    def reset(self):
        """Reset all statistics to zero"""
        self.stats.reset()

    def get_stats(self) -> StatsDict:
        """Return detailed statistics about this DAC"""
        return self.stats.get_stats()
