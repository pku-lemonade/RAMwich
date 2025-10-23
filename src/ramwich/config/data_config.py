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

    activation_width: int = Field(default=8, description="Activation data bits")

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

    @model_validator(mode="after")
    def calculate_derived_values(self):
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

        return self
