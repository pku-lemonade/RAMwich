from typing import ClassVar

from pydantic import BaseModel, Field, model_validator


class DACConfig(BaseModel):
    """Digital-to-Analog Converter configuration"""

    # Class constants for lookup tables
    LAT_DICT: ClassVar[dict[int, float]] = {1: 0.034746, 2: 0.034746, 4: 0.034746, 8: 0.034746, 16: 0.034746}
    POW_DYN_DICT: ClassVar[dict[int, float]] = {1: 1.144483, 2: 1.144483, 4: 1.144483, 8: 1.144483, 16: 1.144483}
    POW_LEAK_DICT: ClassVar[dict[int, float]] = {1: 0.0, 2: 0.0, 4: 0.0, 8: 0.0, 16: 0.0}
    AREA_DICT: ClassVar[dict[int, float]] = {1: 0.000275, 2: 0.000305, 4: 0.000356, 8: 0.000489, 16: 0.000733}

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
