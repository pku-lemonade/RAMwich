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
    LAT_DICT: ClassVar[dict[int, float]] = {1: 2, 2: 3, 4: 5, 8: 9, 16: 17}
    POW_DYN_DICT: ClassVar[dict[int, float]] = {1: 0.8592847, 2: 1.00747422, 4: 1.5334067, 8: 2.5617338, 16: 5.0623805}
    POW_LEAK_DICT: ClassVar[dict[int, float]] = {1: 0.0, 2: 0.0, 4: 0.0, 8: 0.0, 16: 0.0}
    AREA_DICT: ClassVar[dict[int, float]] = {1: 0.000788, 2: 0.001088, 4: 0.001687, 8: 0.002887, 16: 0.005287}

    resolution: int = Field(default=8, description="ADC resolution")

    lat: float = Field(default=None, init=False, description="ADC latency")
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
