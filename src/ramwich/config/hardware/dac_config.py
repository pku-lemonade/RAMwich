from typing import ClassVar

from pydantic import BaseModel, Field, model_validator


class DACConfig(BaseModel):
    """Digital-to-Analog Converter configuration"""

    # Class constants for lookup tables
    LAT_DICT: ClassVar[dict[int, int]] = {1: 1, 2: 1, 4: 1, 8: 1, 16: 1}
    POW_DYN_DICT: ClassVar[dict[int, float]] = {
        1: 0.5739,
        2: 1.1444,
        4: 2.4808,
        8: 4.0827,
        16: 6.6344,
    }
    POW_LEAK_DICT: ClassVar[dict[int, float]] = {
        1: 0,
        2: 0,
        4: 0,
        8: 0,
        16: 0,
    }
    AREA_DICT: ClassVar[dict[int, float]] = {1: 0.000466, 2: 0.000489, 4: 0.000601, 8: 0.000703, 16: 0.000915}

    resolution: int = Field(default=1, description="DAC resolution")
    VDD: float = Field(default=1, description="Supply voltage")

    lat: int = Field(default=None, init=False, description="DAC latency")
    pow_dyn: float = Field(default=None, init=False, description="DAC dynamic power")
    pow_leak: float = Field(default=None, init=False, description="DAC leakage power")
    area: float = Field(default=None, init=False, description="DAC area")

    @model_validator(mode="after")
    def calculate_derived_values(self):
        # Update derived values if resolution is different from default
        if self.resolution in self.LAT_DICT:
            self.lat = int(self.LAT_DICT[self.resolution])
            self.pow_dyn = self.POW_DYN_DICT[self.resolution]
            self.pow_leak = self.POW_LEAK_DICT[self.resolution]
            self.area = self.AREA_DICT[self.resolution]

        return self
