import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..data_config import BitConfig, DataConfig
from ..hardware.adc_config import ADCConfig
from ..hardware.dac_config import DACConfig
from ..hardware.xbar_config import XBARConfig


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
    mux_pow_leak: float = Field(default=0.001, description="Single MUX leakage power")
    mux_pow_dyn: float = Field(default=0.01893, description="Single MUX dynamic power")
    mux_area: float = Field(default=0.000005, description="Single MUX area")

    sna_lat: float = Field(default=1, description="Single shift and adder processing latency")
    sna_pow_leak: float = Field(default=0.001, description="Single shift and adder leakage power")
    sna_pow_dyn: float = Field(default=0.158, description="Single shift and adder dynamic power")
    sna_area: float = Field(default=0.000031, description="Single shift and adder area")

    num_columns_per_adc: int = Field(default=16, description="Number of columns per ADC")
    num_adc_per_xbar: int = Field(default=None, init=False, description="Number of ADCs per crossbar")

    num_columns_per_calculator: int = Field(default=128, description="Number of columns per SRAM CIM calculator")
    num_calculator_per_xbar: int = Field(
        default=None, init=False, description="Number of SRAM CIM calculators per crossbar"
    )

    dac_config: DACConfig = Field(default_factory=DACConfig)
    xbar_config: XBARConfig = Field(default_factory=XBARConfig)
    adc_config: ADCConfig = Field(default_factory=ADCConfig)

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

    num_iterations: int = Field(default=None, init=False, description="Number of iterations for one forward pass")

    # Precomputed SRAM-specific typing and indices (to avoid recomputation per unit)
    sram_xbar_types: list[str] = Field(
        default_factory=list, init=False, description="Type per SRAM xbar: INT/EXPW/EXPA/MANT"
    )
    sram_mvm_indices: list[int] = Field(
        default_factory=list, init=False, description="SRAM xbars needing linear MVM (INT,MANT)"
    )
    sram_ewmvm_indices: list[int] = Field(
        default_factory=list, init=False, description="SRAM xbars needing exp weight MVM (EXPW,MANT)"
    )
    sram_eaa_indices: list[int] = Field(
        default_factory=list, init=False, description="SRAM xbars needing exp activation accumulate (EXPA,MANT)"
    )
    sram_mant_indices: list[int] = Field(
        default_factory=list, init=False, description="SRAM xbars needing MANT processing (MANT only)"
    )
    expw_partition_indices: list[list[int]] = Field(
        default_factory=list,
        init=False,
        description="Bit indices grouped per EXPW partition",
    )

    @model_validator(mode="after")
    def calculate_derived_values(self):
        self.num_adc_per_xbar = self.xbar_config.xbar_size // self.num_columns_per_adc

        # Then verify it's a clean division
        if self.xbar_config.xbar_size % self.num_columns_per_adc != 0:
            raise ValueError(
                f"xbar_size ({self.xbar_config.xbar_size}) must be exactly divisible by "
                f"num_columns_per_adc ({self.num_columns_per_adc})"
            )

        self.num_iterations = int(np.ceil(self.data_config.activation_width / self.dac_config.resolution))

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
        for i in self.data_config.storage_config:
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

        # Propagate weight width into data config and compute its derived values
        self.data_config.weight_width = bits
        # Recompute data-config derived values now that weight_width is known
        self.data_config.calculate_derived_values()

        # Validate that EXP/MANT partitions are not mapped onto RRAM bits
        wp = self.data_config.weight_partition
        pl = self.data_config.part_length
        df = self.data_config.data_format
        for i, fmt in enumerate(df):
            if fmt in ("EXPA", "EXPW", "MANT"):
                for j in range(wp[i], wp[i] + pl[i]):
                    # Bits are indexed from LSB=0; is_bit_rram was built LSB->MSB
                    if self.is_bit_rram[-j - 1]:
                        raise ValueError(f"Data format {fmt} not supported for RRAM bit at position {j}.")

        # Assign output maps based on xbar type
        for i in range(self.num_xbar_per_mvmu):
            if self.is_xbar_rram[i]:
                self.rram_to_output_map.append(i)
            else:
                self.sram_to_output_map.append(i)

        # Precompute SRAM xbar types and masks/indices in SRAM-local order
        self.sram_xbar_types = []
        self.sram_mvm_indices = []
        self.sram_eaa_indices = []
        self.sram_ewmvm_indices = []
        self.sram_mant_indices = []

        if self.have_sram_xbar:
            # Build list of (global_xbar_idx, sram_local_idx)
            sram_local = 0
            for global_idx in range(self.num_xbar_per_mvmu):
                if not self.is_xbar_rram[global_idx]:
                    bit_position = self.stored_bit[global_idx]
                    # Find partition index such that weight_partition[p] <= bit_position
                    part_idx = 0
                    for p_idx, wp in enumerate(self.data_config.weight_partition):
                        if bit_position >= wp:
                            part_idx = p_idx
                    xbar_type = self.data_config.data_format[part_idx]
                    self.sram_xbar_types.append(xbar_type)
                    # Populate indices according to desired behavior
                    if xbar_type in ("INT", "MANT"):
                        self.sram_mvm_indices.append(sram_local)
                    if xbar_type in ("EXPA"):
                        self.sram_eaa_indices.append(sram_local)
                    if xbar_type in ("EXPW", "MANT"):
                        self.sram_ewmvm_indices.append(sram_local)
                    if xbar_type == "MANT":
                        self.sram_mant_indices.append(sram_local)
                    sram_local += 1

        # Build summary of partitions typed as EXPW and MANT
        self.expw_partition_indices = []

        for part_idx, fmt in enumerate(self.data_config.data_format):
            start_bit = self.data_config.weight_partition[part_idx]
            part_len = self.data_config.part_length[part_idx]
            indices = list(range(start_bit, start_bit + part_len))

            if fmt in ["EXPW", "MANT"]:
                self.expw_partition_indices.append(indices)

        return self
