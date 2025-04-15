from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..config import DataConfig, MVMUConfig, XBARConfig, DACConfig, ADCConfig
from ..stats import Stats


class ADCStats(BaseModel):
    """Statistics tracking for ADC (Analog-to-Digital Converter) components"""

    # Basic metrics
    conversions: int = Field(default=0, description="Number of A/D conversions performed")
    overflow_times: int = Field(default=0, description="Number of overflows that happens when converting")
    conversion_errors: float = Field(default=0.0, description="Accumulated conversion error")

    # Performance metrics
    active_cycles: int = Field(default=0, description="Number of active cycles")
    energy_consumption: float = Field(default=0.0, description="Energy consumption in pJ")

    def record_conversion(self, overflow: int = 0, error: float = 0.0):
        """Record an ADC conversion operation"""
        self.conversions += 1
        self.overflow_times += overflow
        self.conversion_errors += error

    def get_stats(self, adc_id: Optional[int] = None) -> Stats:
        """Get ADC-specific statistics"""
        stats = Stats()

        # Map ADC metrics to Stat object
        stats.latency = float(self.active_cycles)
        stats.energy = float(self.energy_consumption)
        stats.area = 0.0  # Set appropriate area value if available

        # Map operation counts
        stats.operations = self.conversions

        return stats


class ADC:
    """Hardware implementation of the ADC component"""

    def __init__(self, mvmu_config: MVMUConfig=None, data_config: DataConfig= None, position: int=None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.data_config = data_config or DataConfig()
        self.adc_config = self.mvmu_config.adc_config
        self.position = position or 0

        # Initialize stats
        self.stats = ADCStats()

        conductance_step = (
            (self.mvmu_config.xbar_config.rram_conductance_max - self.mvmu_config.xbar_config.rram_conductance_min) /
            (2 ** self.data_config.bits_per_cell[self.position] - 1)
        )
        voltage_step = self.mvmu_config.dac_config.VDD / (2 ** self.mvmu_config.dac_config.resolution - 1)
        self.current_step = voltage_step * conductance_step

    def convert(self, analog_value):
        """Simulate ADC conversion from analog to digital"""

        # Calculate max value based on resolution
        max_value = (1 << self.adc_config.resolution) - 1

        # Apply quantization based on resolution
        ideal_value = analog_value / self.current_step
        int_value = int(ideal_value)
        error = ideal_value - int_value

        # Check if overflow occurs
        overflow = 0
        if int_value > max_value:
            int_value = max_value
            overflow = 1

        # Update stats
        self.stats.record_conversion(overflow=overflow, error=error)
        self.stats.active_cycles += self.adc_config.lat
        self.stats.energy_consumption += self.adc_config.pow_dyn

        return int_value

    def get_energy_consumption(self):
        """Return the total energy consumption in pJ"""
        return self.stats.energy_consumption

    def get_stats(self) -> Stats:
        """Return detailed statistics about this ADC"""
        return self.stats.get_stats(self.adc_id)
