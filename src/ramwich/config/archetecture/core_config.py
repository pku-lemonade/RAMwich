import math
from typing import ClassVar
from pydantic import BaseModel, Field, model_validator

class CoreConfig(BaseModel):
    """Core configuration"""

    # Core Control unit (control unit and pipeline registers)
    ccu_pow_dyn: float = Field(default=1.25 * 0.2, description="Core control unit dynamic power")
    ccu_pow_leak: float = Field(default=0, description="Core control unit leakage power")
    ccu_area: float = Field(default=0.00145 * 2.25, description="Core control unit area")

    # Memory lookup tables
    DATA_MEM_LAT_DICT: ClassVar[dict[int, int]] = {256: 1, 512: 1, 1024: 1, 2048: 1, 4096: 1}
    DATA_MEM_POW_DYN_DICT: ClassVar[dict[int, float]] = {256: 0.16, 512: 0.24, 1024: 0.33, 2048: 0.57, 4096: 1}
    DATA_MEM_POW_LEAK_DICT: ClassVar[dict[int, float]] = {256: 0.044, 512: 0.078, 1024: 0.147, 2048: 0.33, 4096: 1}
    DATA_MEM_AREA_DICT: ClassVar[dict[int, float]] = {
        256: 0.00056,
        512: 0.00108,
        1024: 0.00192,
        2048: 0.00392,
        4096: 0.00392,  # Aligned with PUMA
    }

    dataMem_size: int = Field(default=4096, description="Data memory size")
    dataMem_lat: float = Field(default=None, init=False, description="Data memory latency")
    dataMem_pow_dyn: float = Field(default=None, init=False, description="Data memory dynamic power")
    dataMem_pow_leak: float = Field(default=None, init=False, description="Data memory leakage power")
    dataMem_area: float = Field(default=None, init=False, description="Data memory area")

    # Instruction memory lookup tables
    INSTRN_MEM_LAT_DICT: ClassVar[dict[int, int]] = {
        256: 1,
        512: 1,
        1024: 1,
        2048: 1,
        4096: 1,
        8192: 1,
        16384: 1,
        32768: 1,
        65536: 1,
        131072: 1,
    }
    INSTRN_MEM_POW_DYN_DICT: ClassVar[dict[int, float]] = {
        256: 0.16,
        512: 0.24,
        1024: 0.33,
        2048: 0.57,
        4096: 1,
        8192: 1,
        16384: 1,
        32768: 1,
        65536: 1,
        131072: 1,
    }
    INSTRN_MEM_POW_LEAK_DICT: ClassVar[dict[int, float]] = {
        256: 0.044,
        512: 0.078,
        1024: 0.147,
        2048: 0.33,
        4096: 1,
        8192: 1,
        16384: 1,
        32768: 1,
        65536: 1,
        131072: 1,
    }
    INSTRN_MEM_AREA_DICT: ClassVar[dict[int, float]] = {
        256: 0.00056,
        512: 0.00108,
        1024: 0.00192,
        2048: 0.0041,
        4096: 0.0041,
        8192: 0.0041,
        16384: 0.0041,
        32768: 0.0041,
        65536: 0.0041,
        131072: 0.0041,
    }

    instrnMem_size: int = Field(default=131072, description="Core instruction memory size")
    instrnMem_lat: float = Field(default=None, init=False, description="Core instruction memory latency")
    instrnMem_pow_dyn: float = Field(default=None, init=False, description="Core instruction memory dynamic power")
    instrnMem_pow_leak: float = Field(default=None, init=False, description="Core instruction memory leakage power")
    instrnMem_area: float = Field(default=None, init=False, description="Core instruction memory area")

    # VFU parameters with default fields
    alu_lat: int = Field(default=1, description="ALU latency")
    num_alu_per_vfu: int = Field(default=12, description="Number of ALUs per VFU")
    alu_pow_dyn: float = Field(default=2.4 * 32 / 45, description="ALU dynamic power")
    alu_pow_div_dyn: float = Field(default=1.52 * 32 / 45, description="ALU division dynamic power")
    alu_pow_mul_dyn: float = Field(default=0.795 * 32 / 45, description="ALU multiplication dynamic power")
    act_pow_leak: float = Field(default=0.026, description="Activation unit leakage power")
    act_pow_dyn: float = Field(default=0.26 - 0.026, description="Activation unit dynamic power")
    alu_pow_others_dyn: float = Field(default=0.373 * 32 / 45, description="ALU other operations dynamic power")
    alu_pow_leak: float = Field(default=0.27 * 32 / 45, description="ALU leakage power")
    alu_area: float = Field(default=0.00567 * 32 / 45, description="ALU area")
    act_area: float = Field(default=0.0003, description="Activation unit area")

    @model_validator(mode="after")
    def calculate_derived_values(self):
        # Override dataMem parameters based on dataMem_size if it differs from default
        if self.dataMem_size in self.DATA_MEM_LAT_DICT:
            self.dataMem_lat = self.DATA_MEM_LAT_DICT[self.dataMem_size]
            self.dataMem_pow_dyn = self.DATA_MEM_POW_DYN_DICT[self.dataMem_size]
            self.dataMem_pow_leak = self.DATA_MEM_POW_LEAK_DICT[self.dataMem_size]
            self.dataMem_area = self.DATA_MEM_AREA_DICT[self.dataMem_size]

        if self.instrnMem_size in self.INSTRN_MEM_LAT_DICT:
            self.instrnMem_lat = self.INSTRN_MEM_LAT_DICT[self.instrnMem_size]
            self.instrnMem_pow_dyn = self.INSTRN_MEM_POW_DYN_DICT[self.instrnMem_size]
            self.instrnMem_pow_leak = self.INSTRN_MEM_POW_LEAK_DICT[self.instrnMem_size]
            self.instrnMem_area = self.INSTRN_MEM_AREA_DICT[self.instrnMem_size] * math.sqrt(8)  # Aligned with PUMA

        return self