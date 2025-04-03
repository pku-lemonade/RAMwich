from src.component_config import ADCConfig

class AnalogToDigitalConverter:
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

        # Performance tracking
        self.conversion_count = 0
        self.active_cycles = 0
        self.energy_consumption = 0

    def convert(self, analog_value):
        """Simulate ADC conversion from analog to digital"""
        # Calculate the number of cycles needed
        cycles = self.adc_config.lat

        # Update stats
        self.conversion_count += 1
        self.active_cycles += cycles
        self.energy_consumption += self.adc_config.pow_dyn * cycles

        # Apply quantization based on resolution
        max_value = (1 << self.resolution) - 1
        quantized = int(min(max(0, analog_value * max_value), max_value))

        return quantized, cycles

    def get_energy_consumption(self):
        """Return the total energy consumption in pJ"""
        return self.energy_consumption
