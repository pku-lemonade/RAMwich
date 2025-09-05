import re
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class BitConfig(str, Enum):
    """Bit configuration for storage types"""

    SLC = "1"
    MLC = "2"
    TLC = "3"
    QLC = "4"
    SRAM = "s"


class DataConfig(BaseModel):
    """Data type configuration"""

    activation_format: str = Field(default="Q4.4", description="Activation format")
    activation_int_bits: int = Field(default=None, init=False, description="Activation integer bits")
    activation_frac_bits: int = Field(default=None, init=False, description="Activation fractional bits")
    activation_width: int = Field(default=None, init=False, description="Activation data bits")

    storage_config: list[BitConfig] = Field(default=[BitConfig.QLC], description="Storage configuration")
    data_format: list[str] = Field(default=["INT"], description="Data format: INT or EXP or MANT for each part")
    weight_partition: list[int] = Field(
        default=[0],
        description="Weight partitioning, e.g.[0,4] for 4 lower bits as one part, 4 higher bits as another part",
    )
    part_length: list[int] = Field(
        default_factory=list, init=False, description="Length of each part in weight partition"
    )
    weight_width: int = Field(default=None, init=False, description="Weight data bits")
    part_number: int = Field(default=None, init=False, description="Number of parts in weight partition")

    stored_bit: list = Field(default_factory=list, init=False, description="Stored bit positions")
    bits_per_cell: list = Field(default_factory=list, init=False, description="Bits per cell")
    is_xbar_rram: list = Field(default_factory=list, init=False, description="Is crossbar RRAM")
    is_bit_rram: list = Field(default_factory=list, init=False, description="Is each bit RRAM")
    rram_to_output_map: list = Field(default_factory=list, init=False, description="RRAM xbars to output map")
    sram_to_output_map: list = Field(default_factory=list, init=False, description="SRAM xbars to output map")
    have_rram_xbar: bool = Field(default=False, description="Whether have RRAM crossbar or not")
    have_sram_xbar: bool = Field(default=False, description="Whether have SRAM crossbar or not")
    num_rram_xbar_per_mvmu: int = Field(default=None, init=False, description="Number of RRAM xbars per MVMU")
    num_sram_xbar_per_mvmu: int = Field(default=None, init=False, description="Number of SRAM xbars per MVMU")
    num_xbar_per_mvmu: int = Field(default=None, init=False, description="Total number of xbars per MVMU")

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

        self.have_sram_xbar = False  # Reset flag to avoid stale state
        self.have_rram_xbar = False  # Reset flag to avoid stale state
        self.stored_bit = []
        self.bits_per_cell = []
        self.is_xbar_rram = []
        self.is_bit_rram = []
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
                self.is_bit_rram.append(False)
                bits += 1
                self.have_sram_xbar = True
            else:
                self.num_rram_xbar_per_mvmu += 1
                self.bits_per_cell.append(int(i))
                self.is_xbar_rram.append(True)
                for _j in range(int(i)):
                    self.is_bit_rram.append(True)
                bits += int(i)
                self.have_rram_xbar = True
        self.stored_bit.append(bits)

        self.num_xbar_per_mvmu = self.num_sram_xbar_per_mvmu + self.num_rram_xbar_per_mvmu
        self.weight_width = bits

        # Validate weight partition
        wp = self.weight_partition
        ww = self.weight_width
        df = self.data_format

        if wp[0] != 0:
            raise ValueError(f"The first start bit in weight partition must be 0, but got {wp[0]}.")
        if any(wp[i + 1] <= wp[i] for i in range(len(wp) - 1)):
            raise ValueError(f"Weight partition must be in increasing order, but got {wp}.")
        if wp[-1] >= ww - 1:
            raise ValueError(
                f"The last start bit in weight partition must be less than the last bit {ww - 1}, but got {wp[-1]}."
            )

        self.part_length = [wp[i + 1] - wp[i] for i in range(len(wp) - 1)] + [ww - wp[-1]]

        if len(wp) != len(df):
            raise ValueError(f"Weight partition length {len(wp)} must match data format length {len(df)}.")

        valid_formats = {"INT", "EXP", "MANT"}
        for i, fmt in enumerate(df):
            if fmt not in valid_formats:
                raise ValueError(f"Invalid data format: {fmt}. Supported formats are INT, EXP, MANT.")
            if fmt in {"EXP", "MANT"}:
                for j in range(wp[i], wp[i] + self.part_length[i]):
                    if self.is_bit_rram[-j - 1]:
                        raise ValueError(f"Data format {fmt} not supported for RRAM bit at position {j}.")

        # Assign output maps based on xbar type
        for i in range(self.num_xbar_per_mvmu):
            if self.is_xbar_rram[i]:
                self.rram_to_output_map.append(i)
            else:
                self.sram_to_output_map.append(i)

        return self
