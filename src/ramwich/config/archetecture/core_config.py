import math
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator


class CoreConfig(BaseModel):
    """Core configuration"""

    core_type: list[list[int]] = Field(default=None, init=False, description="Core type configuration")

    # Core Control unit (control unit and pipeline registers)
    ccu_pow_dyn: float = Field(default=0.215826, description="Core control unit dynamic power")
    ccu_pow_leak: float = Field(default=0.023201, description="Core control unit leakage power")
    ccu_area: float = Field(default=0.000124416, description="Core control unit area")

    # Memory lookup tables
    DATA_MEM_LAT_DICT: ClassVar[dict[int, int]] = {256: 1, 512: 1, 1024: 1, 2048: 1, 4096: 1, 8192: 1}
    DATA_MEM_POW_DYN_DICT: ClassVar[dict[int, float]] = {256: 0.0014, 512: 0.00188, 1024: 0.00304, 2048: 0.00383, 4096: 0.00925, 8192: 0.0195}
    DATA_MEM_POW_LEAK_DICT: ClassVar[dict[int, float]] = {256: 0.000176, 512: 0.000342, 1024: 0.000614, 2048: 0.001216, 4096: 0.004438, 8192: 0.008698}
    DATA_MEM_AREA_DICT: ClassVar[dict[int, float]] = {256: 0.0606252, 512: 0.0108623, 1024: 0.0171094, 2048: 0.0356512, 4096: 0.136807, 8192: 0.302279}

    dataMem_size: int = Field(default=4096, description="Data memory size")
    dataMem_lat: int = Field(default=None, description="Data memory latency")
    dataMem_pow_dyn: float = Field(default=None, init=False, description="Data memory dynamic power")
    dataMem_pow_leak: float = Field(default=None, init=False, description="Data memory leakage power")
    dataMem_area: float = Field(default=None, init=False, description="Data memory area")

    storage_size: int = Field(default=1024, description="Storage size")
    storage_lat: int = Field(default=None, init=False, description="Storage latency")
    storage_pow_dyn: float = Field(default=None, init=False, description="Storage dynamic power")
    storage_pow_leak: float = Field(default=None, init=False, description="Storage leakage power")
    storage_area: float = Field(default=None, init=False, description="Storage area")

    # Instruction memory lookup tables
    INSTRN_MEM_LAT_DICT: ClassVar[dict[int, int]] = {
        256: 2,
        512: 2,
        1024: 2,
        2048: 2,
        4096: 2,
        8192: 2,
        16384: 2,
        32768: 2,
        65536: 2,
        131072: 3,
    }
    INSTRN_MEM_POW_DYN_DICT: ClassVar[dict[int, float]] = {
        256: 0.01623,
        512: 0.01726,
        1024: 0.018365,
        2048: 0.02168,
        4096: 0.02781,
        8192: 0.02825,
        16384: 0.02596,
        32768: 0.03351,
        65536: 0.03981,
        131072: 0.043195,
    }
    INSTRN_MEM_POW_LEAK_DICT: ClassVar[dict[int, float]] = {
        256: 0.000377,
        512: 0.000411,
        1024: 0.000603,
        2048: 0.001097,
        4096: 0.001982,
        8192: 0.003984,
        16384: 0.008845,
        32768: 0.017154,
        65536: 0.031180,
        131072: 0.62359,
    }
    INSTRN_MEM_AREA_DICT: ClassVar[dict[int, float]] = {
        256: 0.0969725,
        512: 0.0723594,
        1024: 0.0769046,
        2048: 0.0786408,
        4096: 0.0740648,
        8192: 0.147075,
        16384: 0.326787,
        32768: 0.521506,
        65536: 0.890833,
        131072: 1.77083,
    }

    instrnMem_size: int = Field(default=131072, description="Core instruction memory size")
    instrnMem_lat: int = Field(default=None, init=False, description="Core instruction memory latency")
    instrnMem_pow_dyn: float = Field(default=None, init=False, description="Core instruction memory dynamic power")
    instrnMem_pow_leak: float = Field(default=None, init=False, description="Core instruction memory leakage power")
    instrnMem_area: float = Field(default=None, init=False, description="Core instruction memory area")

    # VFU parameters with default fields

    # ALU Unit(IVFU)
    alu_lat: int = Field(default=10, description="ALU latency")
    num_alu_per_ivfu: int = Field(default=12, description="Number of ALUs per VFU")
    alu_pow_dyn: float = Field(default=1.7413, description="ALU dynamic power")
    alu_pow_div_dyn: float = Field(default=0.8008, description="ALU division dynamic power")
    alu_pow_mul_dyn: float = Field(default=0.6765, description="ALU multiplication dynamic power")
    alu_pow_others_dyn: float = Field(default=0.264, description="ALU other operations dynamic power")
    alu_pow_leak: float = Field(default=0.0445, description="ALU leakage power")
    alu_pow_leak_mul: float = Field(default=0.0114151, description="ALU multiplication leakage power")
    alu_pow_leak_div: float = Field(default=0.0294305, description="ALU division leakage power")
    alu_pow_leak_others: float = Field(default=0.005, description="ALU other operations leakage power")
    alu_area: float = Field(default=0.001601, description="ALU area")

    # Activation Unit

    act_area: float = Field(default=0.000023068, description="Activation unit area")
    act_pow_leak: float = Field(default=4.647309, description="Activation unit leakage power")
    act_pow_dyn: float = Field(default=0.065739, description="Activation unit dynamic power")

    # FVFU Unit
    fvfu_lat: int = Field(default=10, description="FVFU latency")
    fvfu_pow_dyn: float = Field(default=1.624453, description="FVFU dynamic power")
    fvfu_pow_mul_dyn: float = Field(default=0.675600, description="FVFU MUL dynamic power")
    fvfu_pow_div_dyn: float = Field(default=0.627092, description="FVFU DIV dynamic power")
    fvfu_pow_others_dyn: float = Field(default=0.533418, description="FVFU other ops dynamic power")
    fvfu_pow_convert_dyn: float = Field(default=0.036746, description="FVFU convert ops dynamic power")
    fvfu_pow_mul_leak: float = Field(default=0.117625, description="FVFU MUL leakage power")
    fvfu_pow_div_leak: float = Field(default=0.072116, description="FVFU DIV leakage power")
    fvfu_pow_convert_leak: float = Field(default=0.036746, description="FVFU convert ops leakage power")
    fvfu_pow_others_leak: float = Field(default=0.076035, description="FVFU other ops leakage power")
    fvfu_pow_leak: float = Field(default=0.248179, description="FVFU leakage power")
    fvfu_area: float = Field(default=0.00161333, description="FVFU area")

    int_min_value: int = Field(default=-(2**7), description="Minimum integer value for INT operations")
    int_max_value: int = Field(default=2**7 - 1, description="Maximum integer value for INT operations")

    @model_validator(mode="after")
    def calculate_derived_values(self):
        # Override dataMem parameters based on dataMem_size if it differs from default
        if self.dataMem_size in self.DATA_MEM_LAT_DICT:
            self.dataMem_lat = self.DATA_MEM_LAT_DICT[self.dataMem_size]
            self.dataMem_pow_dyn = self.DATA_MEM_POW_DYN_DICT[self.dataMem_size]
            self.dataMem_pow_leak = self.DATA_MEM_POW_LEAK_DICT[self.dataMem_size]
            self.dataMem_area = self.DATA_MEM_AREA_DICT[self.dataMem_size]

        if self.storage_size in self.DATA_MEM_LAT_DICT:
            self.storage_lat = self.DATA_MEM_LAT_DICT[self.storage_size]
            self.storage_pow_dyn = self.DATA_MEM_POW_DYN_DICT[self.storage_size]
            self.storage_pow_leak = self.DATA_MEM_POW_LEAK_DICT[self.storage_size]
            self.storage_area = self.DATA_MEM_AREA_DICT[self.storage_size]

        if self.instrnMem_size in self.INSTRN_MEM_LAT_DICT:
            self.instrnMem_lat = self.INSTRN_MEM_LAT_DICT[self.instrnMem_size]
            self.instrnMem_pow_dyn = self.INSTRN_MEM_POW_DYN_DICT[self.instrnMem_size]
            self.instrnMem_pow_leak = self.INSTRN_MEM_POW_LEAK_DICT[self.instrnMem_size]
            self.instrnMem_area = self.INSTRN_MEM_AREA_DICT[self.instrnMem_size] * math.sqrt(8)  # Aligned with PUMA

        return self
