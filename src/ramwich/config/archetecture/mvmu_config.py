from typing import ClassVar
from pydantic import BaseModel, Field, model_validator
from ..hardware.adc_config import ADCConfig
from ..hardware.dac_config import DACConfig
from ..hardware.xbar_config import XBARConfig

class MVMUConfig(BaseModel):
    """Matrix-Vector Multiply Unit configuration"""

    snh_lat: float = Field(default=1, description="Single sample and holder processing latency")
    snh_pow_leak: float = Field(default=9.7 * 10 ** (-7), description="Single sample and holder leakage power")
    snh_pow_dyn: float = Field(
        default=9.7 * 10 ** (-6) - 9.7 * 10 ** (-7), description="Single sample and holder dynamic power"
    )
    snh_area: float = Field(default=0.00004 / 8 / 128, description="Single sample and holder area")

    mux_lat: float = Field(default=0, description="Single MUX processing latency")
    mux_pow_leak: float = Field(default=0, description="Single MUX leakage power")
    mux_pow_dyn: float = Field(default=0, description="Single MUX dynamic power")
    mux_area: float = Field(default=0, description="Single MUX area")

    sna_lat: float = Field(default=1, description="Single shift and adder processing latency")
    sna_pow_leak: float = Field(default=0.005, description="Single shift and adder leakage power")
    sna_pow_dyn: float = Field(default=0.05 - 0.005, description="Single shift and adder dynamic power")
    sna_area: float = Field(default=0.00006, description="Single shift and adder area")

    num_columns_per_adc: int = Field(default=16, description="Number of columns per ADC")
    num_adc_per_xbar: int = Field(default=None, init=False, description="Number of ADCs per crossbar")

    num_columns_per_calculator: int = Field(default=128, description="Number of columns per SRAM CIM calculator")
    num_calculator_per_xbar: int = Field(
        default=None, init=False, description="Number of SRAM CIM calculators per crossbar"
    )

    num_rram_xbar_per_mvmu: int = Field(default=None, init=False, description="Number of RRAM xbars")
    num_sram_xbar_per_mvmu: int = Field(default=None, init=False, description="Number of SRAM xbars")
    num_xbar_per_mvmu: int = Field(default=None, init=False, description="Number of crossbars per MVMU")

    stored_bit: list = Field(default=None, init=False, description="Stored bit positions")
    bits_per_cell: list = Field(default=None, init=False, description="Bits per cell")
    is_xbar_rram: list = Field(default=None, init=False, description="Is crossbar RRAM")
    rram_to_output_map: list = Field(default=None, init=False, description="RRAM xbars to output map")
    sram_to_output_map: list = Field(default=None, init=False, description="SRAM xbars to output map")
    have_rram_xbar: bool = Field(default=False, description="Whether have RRAM crossbar or not")
    have_sram_xbar: bool = Field(default=False, description="Whether have SRAM crossbar or not")

    dac_config: DACConfig = Field(default_factory=DACConfig)
    xbar_config: XBARConfig = Field(default_factory=XBARConfig)
    adc_config: ADCConfig = Field(default_factory=ADCConfig)
        
    @model_validator(mode="after")
    def calculate_derived_values(self):
        self.num_adc_per_xbar = self.xbar_config.xbar_size // self.num_columns_per_adc

        # Then verify it's a clean division
        if self.xbar_config.xbar_size % self.num_columns_per_adc != 0:
            raise ValueError(
                f"xbar_size ({self.xbar_config.xbar_size}) must be exactly divisible by "
                f"num_columns_per_adc ({self.num_columns_per_adc})"
            )
        
        return self