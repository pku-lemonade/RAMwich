from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import ADCType, MVMUConfig
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

    def __init__(self, mvmu_config: MVMUConfig = None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.adc_config = self.mvmu_config.adc_config

        # calculate max value based on resolution
        self.max_value = (1 << self.adc_config.resolution) - 1
        self.min_value = -self.max_value

        # calculate size of ADC array
        self.shape = (self.mvmu_config.num_rram_xbar_per_mvmu, self.mvmu_config.num_adc_per_xbar)
        self.size = np.prod(self.shape)

        if self.adc_config.type == ADCType.DIFFERENTIAL:
            self.size *= 1
        else:
            self.size *= 2

        # Calculate current step for each ADC
        # Create array mapping each ADC to its corresponding xbar
        xbar_indices = np.where(self.mvmu_config.is_xbar_rram)[0]

        # Vectorized calculation of conductance steps
        xbar_bits = np.array([self.mvmu_config.bits_per_cell[idx] for idx in xbar_indices])[:, np.newaxis]

        voltage_step = self.mvmu_config.dac_config.VDD / (2**self.mvmu_config.dac_config.resolution - 1)
        conductance_range = (
            self.mvmu_config.xbar_config.rram_conductance_max - self.mvmu_config.xbar_config.rram_conductance_min
        )
        conductance_steps = conductance_range / ((2**xbar_bits) - 1)

        # Calculate current step for each ADC
        self.current_step = voltage_step * conductance_steps

        # Initialize stats
        self.stats = ADCStats()

    def convert(self, analog_value_pos: NDArray[np.float64], analog_value_neg: NDArray[np.float64]):
        """Simulate ADC conversion from analog to digital"""

        # Validate input
        if analog_value_pos.shape != analog_value_neg.shape:
            raise ValueError(
                f"Expected input vectors of the same shape, got {analog_value_pos.shape} and {analog_value_neg.shape}"
            )

        if analog_value_pos.shape != self.shape:
            raise ValueError(f"Expected input vectors of shape {self.shape}, got {analog_value_pos.shape}")

        # Apply quantization based on resolution
        ideal_values_pos = analog_value_pos / self.current_step
        ideal_values_neg = analog_value_neg / self.current_step
        ideal_values = ideal_values_pos - ideal_values_neg
        int_values_pos = np.round(ideal_values_pos).astype(np.int_)
        int_values_neg = np.round(ideal_values_neg).astype(np.int_)
        int_values = int_values_pos - int_values_neg
        errors = np.abs(ideal_values - int_values)

        # Check if overflow occurs
        overflow_mask = (int_values > self.max_value) | (int_values < self.min_value)
        overflow_count = np.sum(overflow_mask)

        # Calculate total error
        total_error = np.sum(errors)

        # Clip values to max_value
        int_values = np.clip(int_values, self.min_value, self.max_value)

        # Update stats
        self.stats.conversions += self.size
        self.stats.overflow_times += overflow_count
        self.stats.conversion_errors += total_error
        self.stats.active_cycles += self.adc_config.lat
        self.stats.energy_consumption += self.adc_config.pow_dyn * self.size

        return int_values

    def get_energy_consumption(self):
        """Return the total energy consumption in pJ"""
        return self.stats.energy_consumption

    def get_stats(self) -> Stats:
        """Return detailed statistics about this ADC"""
        return self.stats.get_stats(self.adc_id)
