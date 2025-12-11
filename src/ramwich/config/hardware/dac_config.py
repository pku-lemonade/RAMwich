from typing import ClassVar

from pydantic import BaseModel, Field, model_validator


class DACConfig(BaseModel):
    """Digital-to-Analog Converter configuration"""

    # Class constants for lookup tables
    LAT_DICT: ClassVar[dict[int, int]] = {1: 1, 2: 1, 4: 1, 8: 1, 16: 1}
    POW_DYN_DICT: ClassVar[dict[int, float]] = {
        1: 5.604e-4,
        2: 1.112e-3,
        4: 2.422e-3,
        8: 3.987e-3,
        16: 6.479e-3,
    }
    POW_LEAK_DICT: ClassVar[dict[int, float]] = {
        1: 0.00078,
        2: 0.00078,
        4: 0.00078,
        8: 0.00078,
        16: 0.00078,
    }
    AREA_DICT: ClassVar[dict[int, float]] = {1: 4.551e-7, 2: 4.775e-7, 4: 5.892e-7, 8: 6.865e-7, 16: 8.935e-7}

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
            self.lat = self.LAT_DICT[self.resolution]
            self.pow_dyn = self.POW_DYN_DICT[self.resolution]
            self.pow_leak = self.POW_LEAK_DICT[self.resolution]
            self.area = self.AREA_DICT[self.resolution]

        return self
