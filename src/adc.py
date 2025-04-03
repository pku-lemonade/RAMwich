from .config import ADCConfig
from typing import Dict, Any, Optional
from pydantic import Field, BaseModel
from .stats import Stats

class ADCStats(BaseModel):
    """Statistics tracking for ADC (Analog-to-Digital Converter) components"""

    # Basic metrics
    conversions: int = Field(default=0, description="Number of A/D conversions performed")
    samples_processed: int = Field(default=0, description="Total number of analog samples processed")
    out_of_range_samples: int = Field(default=0, description="Number of samples that were out of ADC range")
    quantization_errors: float = Field(default=0.0, description="Accumulated quantization error")

    # Performance metrics
    active_cycles: int = Field(default=0, description="Number of active cycles")
    energy_consumption: float = Field(default=0.0, description="Energy consumption in pJ")

    def record_conversion(self, num_samples: int = 1, out_of_range: int = 0, quant_error: float = 0.0):
        """Record an ADC conversion operation"""
        self.conversions += 1
        self.samples_processed += num_samples
        self.out_of_range_samples += out_of_range
        self.quantization_errors += quant_error

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

    def __init__(self, adc_config=None, adc_id=0):
        self.adc_config = adc_config if adc_config else ADCConfig()
        self.adc_id = adc_id

        # Check if this ADC has a specific resolution setting
        matrix_key = f'matrix_adc_{adc_id}'
        if matrix_key in self.adc_config.res_new:
            self.resolution = self.adc_config.res_new[matrix_key]
        else:
            self.resolution = self.adc_config.resolution

        # Initialize stats
        self.stats = ADCStats()

    def convert(self, analog_value):
        """Simulate ADC conversion from analog to digital"""
        # Calculate the number of cycles needed
        cycles = self.adc_config.lat

        # Calculate max value based on resolution
        max_value = (1 << self.resolution) - 1

        # Check if value is out of range
        out_of_range = 0
        if analog_value < 0 or analog_value > 1:
            out_of_range = 1

        # Apply quantization based on resolution
        capped_value = min(max(0, analog_value), 1)
        ideal_value = capped_value * max_value
        quantized = int(ideal_value)
        quant_error = abs(ideal_value - quantized)

        # Update stats
        self.stats.record_conversion(num_samples=1, out_of_range=out_of_range, quant_error=quant_error)
        self.stats.active_cycles += cycles
        self.stats.energy_consumption += self.adc_config.pow_dyn * cycles

        return quantized, cycles

    def get_energy_consumption(self):
        """Return the total energy consumption in pJ"""
        return self.stats.energy_consumption

    def get_stats(self) -> Stats:
        """Return detailed statistics about this ADC"""
        return self.stats.get_stats(self.adc_id)
