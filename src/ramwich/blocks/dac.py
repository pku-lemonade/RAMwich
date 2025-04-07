from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..config import DACConfig
from ..stats import Stat


class DACStats(BaseModel):
    """Statistics tracking for DAC (Digital-to-Analog Converter) components"""

    # Basic metrics
    conversions: int = Field(default=0, description="Number of D/A conversions performed")
    digital_values_processed: int = Field(default=0, description="Total number of digital values processed")
    zero_values: int = Field(default=0, description="Number of zero values processed")
    max_values: int = Field(default=0, description="Number of maximum values processed")

    # Performance metrics
    active_cycles: int = Field(default=0, description="Number of active cycles")
    energy_consumption: float = Field(default=0.0, description="Energy consumption in pJ")

    def record_conversion(self, digital_value: int, max_digital_value: int):
        """Record a DAC conversion operation"""
        self.conversions += 1
        self.digital_values_processed += 1

        # Track range statistics
        if digital_value == 0:
            self.zero_values += 1
        elif digital_value >= max_digital_value:
            self.max_values += 1

    def get_stats(self, dac_id: Optional[int] = None) -> Stat:
        """Get DAC-specific statistics"""
        stats = Stat()

        # Map DAC metrics to Stat object
        stats.latency = float(self.active_cycles)
        stats.energy = float(self.energy_consumption)
        stats.area = 0.0  # Set appropriate area value if available

        # Map operation counts
        stats.operations = self.conversions

        # Set execution time metrics
        stats.total_execution_time = float(self.active_cycles)

        return stats


class DAC:
    """Hardware implementation of the DAC component"""

    def __init__(self, dac_config=None, dac_id=0):
        self.dac_config = dac_config if dac_config else DACConfig()
        self.resolution = self.dac_config.resolution
        self.dac_id = dac_id

        # Initialize stats
        self.stats = DACStats()

    def convert(self, digital_value):
        """Simulate DAC conversion from digital to analog"""
        # Calculate the number of cycles needed
        cycles = self.dac_config.lat

        # Calculate max value based on resolution
        max_value = (1 << self.resolution) - 1

        # Apply analog conversion based on resolution
        capped_value = min(max(0, digital_value), max_value)
        analog = capped_value / max_value

        # Update stats
        self.stats.record_conversion(digital_value, max_value)
        self.stats.active_cycles += cycles
        self.stats.energy_consumption += self.dac_config.pow_dyn * cycles

        return analog, cycles

    def get_energy_consumption(self):
        """Return the total energy consumption in pJ"""
        return self.stats.energy_consumption

    def get_stats(self) -> Stat:
        """Return detailed statistics about this DAC"""
        return self.stats.get_stats(self.dac_id)
