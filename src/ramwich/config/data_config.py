import re
from enum import Enum
from typing import ClassVar
from pydantic import BaseModel, Field, model_validator


class BitConfig(str, Enum):
    SLC = "1"  # 单层单元
    MLC = "2"  # 多层单元
    TLC = "3"  # 三层单元
    QLC = "4"  # 四层单元
    SRAM = "s"  # SRAM类型


class DataConfig(BaseModel):
    """Data type configuration"""

    storage_config: list[BitConfig] = Field(
        default=[BitConfig.MLC, BitConfig.MLC, BitConfig.MLC, BitConfig.MLC], description="Storage configuration"
    )
    weight_format: str = Field(default="Q1.7", description="Weight format")
    weight_int_bits: int = Field(default=None, init=False, description="Weight integer bits")
    weight_frac_bits: int = Field(default=None, init=False, description="Weight fractional bits")
    weight_width: int = Field(default=None, init=False, description="Weight data bits")

    activation_format: str = Field(default="Q4.4", description="Activation format")
    activation_int_bits: int = Field(default=None, init=False, description="Activation integer bits")
    activation_frac_bits: int = Field(default=None, init=False, description="Activation fractional bits")
    activation_width: int = Field(default=None, init=False, description="Activation data bits")

    addr_width: int = Field(default=32, description="Address width")
    instrn_width: int = Field(default=48, description="Instruction width")

    @model_validator(mode="after")
    def calculate_derived_values(self):
        # Calculate weight and activation bits based on the provided formats
        pattern = r"Q(\d+)\.(\d+)"

        match = re.match(pattern, self.weight_format)
        if match:
            self.weight_int_bits = int(match.group(1))
            self.weight_frac_bits = int(match.group(2))
            self.weight_width = self.weight_int_bits + self.weight_frac_bits
        else:
            raise ValueError(f"Invalid weight format: {self.weight_format}")

        match = re.match(pattern, self.activation_format)
        if match:
            self.activation_int_bits = int(match.group(1))
            self.activation_frac_bits = int(match.group(2))
            self.activation_width = self.activation_int_bits + self.activation_frac_bits
        else:
            raise ValueError(f"Invalid activation format: {self.activation_format}")

        return self