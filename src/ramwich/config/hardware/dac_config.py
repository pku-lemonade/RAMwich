from typing import ClassVar
from pydantic import BaseModel, Field, model_validator

class DACConfig(BaseModel):
    """Digital-to-Analog Converter configuration"""

    # Class constants for lookup tables
    LAT_DICT: ClassVar[dict[int, int]] = {1: 1, 2: 1, 4: 1, 8: 1, 16: 1}
    POW_DYN_DICT: ClassVar[dict[int, float]] = {
        1: 0.00350625,
        2: 0.00350625,
        4: 0.00350625,
        8: 0.00350625,
        16: 0.00350625,
    }
    POW_LEAK_DICT: ClassVar[dict[int, float]] = {
        1: 0.000390625,
        2: 0.000390625,
        4: 0.000390625,
        8: 0.000390625,
        16: 0.000390625,
    }
    AREA_DICT: ClassVar[dict[int, float]] = {1: 1.67e-7, 2: 1.67e-7, 4: 1.67e-7, 8: 1.67e-7, 16: 1.67e-7}

    resolution: int = Field(default=1, description="DAC resolution")
    VDD: float = Field(default=1, description="Supply voltage")

    lat: float = Field(default=None, init=False, description="DAC latency")
    pow_dyn: float = Field(default=None, init=False, description="DAC dynamic power")
    pow_leak: float = Field(default=None, init=False, description="DAC leakage power")
    area: float = Field(default=None, init=False, description="DAC area")

    @model_validator(mode="after")
    def calculate_derived_values(self):
        # Update derived values if resolution is different from default
        if self.resolution in self.LAT_DICT:
            self.lat = self.LAT_DICT[self.resolution]
            self.pow_dyn = self.POW_DYN_DICT[self.resolution]
            self.pow_leak = self.POW_LEAK_DICT[self.resolution]
            self.area = self.AREA_DICT[self.resolution]
            
        return self