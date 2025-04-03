from .config import DACConfig
from typing import Dict, Any, Optional
from pydantic import Field, BaseModel

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

    def get_stats(self, dac_id: Optional[int] = None) -> Dict[str, Any]:
        """Get DAC-specific statistics"""
        result = {
            'stats': {
                'conversions': self.conversions,
                'digital_values_processed': self.digital_values_processed,
                'zero_values': self.zero_values,
                'max_values': self.max_values,
                'active_cycles': self.active_cycles,
                'energy_consumption': self.energy_consumption,
            }
        }

        # Add DAC-specific derived metrics
        if self.digital_values_processed > 0:
            result['stats']['zero_value_percentage'] = (self.zero_values / self.digital_values_processed) * 100
            result['stats']['max_value_percentage'] = (self.max_values / self.digital_values_processed) * 100

        if dac_id is not None:
            result['dac_id'] = dac_id

        return result

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

    def get_stats(self):
        """Return detailed statistics about this DAC"""
        return self.stats.get_stats(self.dac_id)
