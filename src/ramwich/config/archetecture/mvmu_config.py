from enum import Enum

from pydantic import BaseModel, Field, model_validator

from ..data_config import DataConfig
from ..hardware.adc_config import ADCConfig
from ..hardware.dac_config import DACConfig
from ..hardware.xbar_config import XBARConfig


class BitConfig(str, Enum):
    """Bit configuration for storage types"""

    SLC = "1"
    MLC = "2"
    TLC = "3"
    QLC = "4"
    SRAM = "s"


class MVMUConfig(BaseModel):
    """Matrix-Vector Multiply Unit configuration"""

    mvmu_type: int = Field(default=0, description="Type of MVMU (0: I/O, 1 to 9: different types)")

    data_config: DataConfig = Field(default_factory=DataConfig)

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

        self.weight_partition = [1, 7]
        self.weight_width = sum(self.weight_partition)

        self.have_sram_xbar = False  # Reset flag to avoid stale state
        self.have_rram_xbar = False  # Reset flag to avoid stale state
        self.stored_bit = []
        self.bits_per_cell = []
        self.is_xbar_rram = []
        self.rram_to_output_map = []
        self.sram_to_output_map = []

        bits = 0  # total bits number in the operand
        self.num_rram_xbar_per_mvmu = 0  # number of RRAM xbars
        self.num_sram_xbar_per_mvmu = 0  # number of SRAM xbars
        for i in self.storage_config:
            self.stored_bit.append(bits)
            if i == BitConfig.SRAM:
                self.num_sram_xbar_per_mvmu += 1
                self.bits_per_cell.append(1)
                self.is_xbar_rram.append(False)
                bits += 1
                self.have_sram_xbar = True
            else:
                self.num_rram_xbar_per_mvmu += 1
                self.bits_per_cell.append(int(i))
                self.is_xbar_rram.append(True)
                bits += int(i)
                self.have_rram_xbar = True
        self.stored_bit.append(bits)
        assert bits == self.weight_width, "storage config invalid: check if total bits in storage config = weight width"

        self.num_xbar_per_mvmu = self.num_sram_xbar_per_mvmu + self.num_rram_xbar_per_mvmu

        # Assign output maps based on xbar type
        for i in range(self.num_xbar_per_mvmu):
            if self.is_xbar_rram[i]:
                self.rram_to_output_map.append(i)
            else:
                self.sram_to_output_map.append(i)

        return self
