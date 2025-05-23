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
    LAT_DICT: ClassVar[dict[int, int]] = {1: 13, 2: 25, 4: 50, 8: 100, 16: 200}
    POW_DYN_DICT: ClassVar[dict[int, float]] = {1: 0.225, 2: 0.45, 4: 0.9, 8: 1.8, 16: 3.2}
    POW_LEAK_DICT: ClassVar[dict[int, float]] = {1: 0.025, 2: 0.05, 4: 0.1, 8: 0.2, 16: 0.4}
    AREA_DICT: ClassVar[dict[int, float]] = {1: 0.0012, 2: 0.0012, 4: 0.0012, 8: 0.0012, 16: 0.0012}

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