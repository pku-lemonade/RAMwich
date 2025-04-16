from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import DataConfig, MVMUConfig, XBARConfig, DACConfig, ADCConfig, ADCType
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


class ADCArray:
    """Hardware implementation of the ADC component"""

    def __init__(self, mvmu_config: MVMUConfig=None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.adc_config = self.mvmu_config.adc_config

        # calculate max value based on resolution
        self.max_value = (1 << self.adc_config.resolution) - 1

        # Calculate current step for each ADC
        num_adc_per_xbar = self.mvmu_config.xbar_config.xbar_size // self.mvmu_config.num_columns_per_adc
        if self.adc_config.type == ADCType.DIFFERENTIAL:
            num_adc_per_xbar *= 1
        else:
            num_adc_per_xbar *= 2
        
        self.size = self.mvmu_config.num_rram_xbar_per_mvmu * num_adc_per_xbar

        # Create array mapping each ADC to its corresponding xbar
        xbar_indices = np.array([i // num_adc_per_xbar for i in range(self.size)])

        # Vectorized calculation of conductance steps
        xbar_bits = np.array([self.mvmu_config.bits_per_cell[idx] for idx in xbar_indices])

        voltage_step = self.mvmu_config.dac_config.VDD / (2 ** self.mvmu_config.dac_config.resolution - 1)
        conductance_range = self.mvmu_config.xbar_config.rram_conductance_max - self.mvmu_config.xbar_config.rram_conductance_min
        conductance_steps = conductance_range / ((2 ** xbar_bits) - 1)

        # Calculate current step for each ADC
        self.current_step = voltage_step * conductance_steps

        # Initialize stats
        self.stats = ADCStats()

    def convert(self, analog_value: NDArray[np.floating]):
        """Simulate ADC conversion from analog to digital"""

        # Validate input
        if analog_value.ndim != 1:
            raise ValueError(f"Expected 1D array, got {analog_value.ndim}D array")

        if len(analog_value) != self.size:
            raise ValueError(f"Expected input vector of shape ({self.size},), got {analog_value.shape}")

        # Apply quantization based on resolution
        ideal_values = analog_value / self.current_step
        int_values = np.floor(ideal_values).astype(np.int_)
        errors = ideal_values - int_values

        # Check if overflow occurs
        overflow_mask = int_values > self.max_value
        overflow_count = np.sum(overflow_mask)

        # Calculate total error
        total_error = np.sum(errors)

        # Clip values to max_value
        int_values = np.clip(int_values, 0, self.max_value)

        # Update stats
        self.stats.record_conversion(overflow=overflow_count, error=total_error)
        self.stats.active_cycles += self.adc_config.lat
        self.stats.energy_consumption += self.adc_config.pow_dyn * self.size

        return int_values

    def get_energy_consumption(self):
        """Return the total energy consumption in pJ"""
        return self.stats.energy_consumption

    def get_stats(self) -> Stats:
        """Return detailed statistics about this ADC"""
        return self.stats.get_stats(self.adc_id)
