from .config import DACConfig

class DAC:
    """Hardware implementation of the DAC component"""

    def __init__(self, dac_config=None):
        self.dac_config = dac_config if dac_config else DACConfig()
        self.resolution = self.dac_config.resolution

        # Performance tracking
        self.conversion_count = 0
        self.active_cycles = 0
        self.energy_consumption = 0

    def convert(self, digital_value):
        """Simulate DAC conversion from digital to analog"""
        # Calculate the number of cycles needed
        cycles = self.dac_config.lat

        # Update stats
        self.conversion_count += 1
        self.active_cycles += cycles
        self.energy_consumption += self.dac_config.pow_dyn * cycles

        # Apply analog conversion based on resolution
        max_value = (1 << self.resolution) - 1
        analog = min(max(0, digital_value), max_value) / max_value

        return analog, cycles

    def get_energy_consumption(self):
        """Return the total energy consumption in pJ"""
        return self.energy_consumption
