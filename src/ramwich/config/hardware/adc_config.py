from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator


class ADCType(str, Enum):
    NORMAL = "normal"
    DIFFERENTIAL = "differential"


class ADCConfig(BaseModel):
    """Analog-to-Digital Converter configuration"""

    type: ADCType = Field(default=ADCType.NORMAL, description="ADC type")

    # Class constants for lookup tables - using integers as keys instead of strings
    LAT_DICT: ClassVar[dict[int, int]] = {1: 10, 2: 10, 4: 10, 5: 6, 8: 9, 16: 10}
    POW_DYN_DICT: ClassVar[dict[int, float]] = {1: 1.9, 2: 1.9, 4: 1.9, 5: 0.0583, 8: 0.1822, 16: 4}
    POW_LEAK_DICT: ClassVar[dict[int, float]] = {1: 0.0, 2: 0.0, 4: 0.0, 5: 0, 8: 0.0, 16: 0.0}
    AREA_DICT: ClassVar[dict[int, float]] = {1: 0.0025, 2: 0.0025, 4: 0.0025, 5: 1.242e-4, 8: 1.804e-4, 16: 0.0055}

    resolution: int = Field(default=8, description="ADC resolution")

    lat: int = Field(default=None, init=False, description="ADC latency")
    pow_dyn: float = Field(default=None, init=False, description="ADC dynamic power")
    pow_leak: float = Field(default=None, init=False, description="ADC leakage power")
    area: float = Field(default=None, init=False, description="ADC area")

    @model_validator(mode="after")
    def calculate_derived_values(self):
        # Update derived values based on resolution if it's different from default
        if self.resolution in self.LAT_DICT:
            self.lat = self.LAT_DICT[self.resolution]
            self.pow_dyn = self.POW_DYN_DICT[self.resolution]
            self.pow_leak = self.POW_LEAK_DICT[self.resolution]
            self.area = self.AREA_DICT[self.resolution]

        return self
