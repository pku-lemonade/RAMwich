import re

from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Data type configuration"""

    activation_format: str = Field(default="Q4.4", description="Activation format")
    activation_int_bits: int = Field(default=None, init=False, description="Activation integer bits")
    activation_frac_bits: int = Field(default=None, init=False, description="Activation fractional bits")
    activation_width: int = Field(default=None, init=False, description="Activation data bits")

    addr_width: int = Field(default=32, description="Address width")
    instrn_width: int = Field(default=48, description="Instruction width")

    @model_validator(mode="after")
    def calculate_derived_values(self):
        # Calculate activation bits based on the provided formats
        pattern = r"Q(\d+)\.(\d+)"

        match = re.match(pattern, self.activation_format)
        if match:
            self.activation_int_bits = int(match.group(1))
            self.activation_frac_bits = int(match.group(2))
            self.activation_width = self.activation_int_bits + self.activation_frac_bits
        else:
            raise ValueError(f"Invalid activation format: {self.activation_format}")

        return self
