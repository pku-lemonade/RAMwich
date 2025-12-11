import math
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator


class TileConfig(BaseModel):
    """Tile configuration"""

    # Tile Control unit
    tcu_pow_dyn: float = Field(default=3.30146, description="Tile control unit dynamic power")
    tcu_pow_leak: float = Field(default=0.381589, description="Tile control unit leakage power")
    tcu_area: float = Field(default=0.001639, description="Tile control unit area")

    # EDRAM lookup tables
    EDRAM_LAT_DICT: ClassVar[dict[int, int]] = {
        8: 3,
        64: 3,
        128: 4,
        2048: 11,
        8192: 18,
        16384: 21,
    }
    EDRAM_POW_DYN_DICT: ClassVar[dict[int, float]] = {
        8: 61.65,
        64: 201.73,
        128: 362.45,
        2048: 363.02,
        8192: 363.48,
        16384: 363.87,
    }

    EDRAM_POW_LEAK_DICT: ClassVar[dict[int, float]] = {8: 0.0, 64: 0.0, 128: 0.0, 2048: 0.0, 8192: 0.02, 16384: 0.03}
    EDRAM_AREA_DICT: ClassVar[dict[int, float]] = {
        8: 0.000348724,
        64: 0.00063015,
        128: 0.000898766,
        2048: 0.0182163,
        8192: 0.0710655,
        16384: 0.153799,
    }

    edram_size_in_KB: int = Field(default=8192, description="EDRAM size in KB")
    edram_size: int = Field(default=4194304, description="EDRAM size")
    edram_lat: int = Field(default=None, init=False, description="EDRAM latency")
    edram_pow_dyn: float = Field(default=None, init=False, description="EDRAM dynamic power")
    edram_pow_leak: float = Field(default=None, init=False, description="EDRAM leakage power")
    edram_area: float = Field(default=None, init=False, description="EDRAM area")

    # Tile instruction memory lookup tables
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
        256: 0.01623,
        512: 0.01726,
        1024: 0.018365,
        2048: 0.02168,
        4096: 0.02781,
        8192: 0.02781,
        16384: 0.02781,
        32768: 0.02781,
        65536: 0.02781,
        131072: 0.02781,
    }
    INSTRN_MEM_POW_LEAK_DICT: ClassVar[dict[int, float]] = {
        256: 0.000377,
        512: 0.000411,
        1024: 0.000603,
        2048: 0.001097,
        4096: 0.001982,
        8192: 0.001982,
        16384: 0.001982,
        32768: 0.001982,
        65536: 0.001982,
        131072: 0.001982,
    }
    INSTRN_MEM_AREA_DICT: ClassVar[dict[int, float]] = {
        256: 0.0969725,
        512: 0.0723594,
        1024: 0.0769046,
        2048: 0.0786408,
        4096: 0.0740648,
        8192: 0.0740648,
        16384: 0.0740648,
        32768: 0.0740648,
        65536: 0.0740648,
        131072: 0.0740648,
    }

    instrnMem_size: int = Field(default=131072, description="Tile instruction memory size")
    instrnMem_lat: int = Field(default=None, init=False, description="Tile instruction memory latency")
    instrnMem_pow_dyn: float = Field(default=None, init=False, description="Tile instruction memory dynamic power")
    instrnMem_pow_leak: float = Field(default=None, init=False, description="Tile instruction memory leakage power")
    instrnMem_area: float = Field(default=None, init=False, description="Tile instruction memory area")

    # EDRAM counter buffer values
    counter_buff_lat: int = Field(default=int(round(1 * math.sqrt(8))), description="Counter buffer latency")
    counter_buff_pow_dyn: float = Field(default=0.65 / 2 * math.sqrt(8), description="Counter buffer dynamic power")
    counter_buff_pow_leak: float = Field(default=0.33 / 2 * math.sqrt(8), description="Counter buffer leakage power")
    counter_buff_area: float = Field(default=0.0041 * math.sqrt(8), description="Counter buffer area")

    # EDRAM to MVMU bus values
    edram_bus_size: int = Field(default=256, description="EDRAM bus size")
    edram_bus_lat: int = Field(default=10, description="EDRAM bus latency")
    edram_bus_pow_dyn: float = Field(default=0.7985, description="EDRAM bus dynamic power")
    edram_bus_pow_leak: float = Field(default=0.0925, description="EDRAM bus leakage power")
    edram_bus_area: float = Field(default=0.0004, description="EDRAM bus area")

    # EDRAM controller values
    edram_ctrl_lat: int = Field(default=8, description="EDRAM controller latency")
    edram_ctrl_pow_dyn: float = Field(default=0.309705, description="EDRAM controller dynamic power")
    edram_ctrl_pow_leak: float = Field(default=0.034536, description="EDRAM controller leakage power")
    edram_ctrl_area: float = Field(default=0.000144581760, description="EDRAM controller area")

    # Receive buffer value dictionary
    receive_buffer_lat: int = Field(default=10, description="Receive buffer latency")
    receive_buffer_pow_dyn: float = Field(default=0.530580, description="Receive buffer dynamic power")
    receive_buffer_pow_leak: float = Field(default=0.056765, description="Receive buffer leakage power")
    receive_buffer_area: float = Field(default=0.000244944002, description="Receive buffer area")

    @model_validator(mode="after")
    def calculate_derived_values(self):
        if self.edram_size_in_KB in self.EDRAM_LAT_DICT:
            self.edram_lat = self.EDRAM_LAT_DICT[self.edram_size_in_KB]
            self.edram_pow_dyn = self.EDRAM_POW_DYN_DICT[self.edram_size_in_KB]
            self.edram_pow_leak = self.EDRAM_POW_LEAK_DICT[self.edram_size_in_KB]
            self.edram_area = self.EDRAM_AREA_DICT[self.edram_size_in_KB]

        if self.instrnMem_size in self.INSTRN_MEM_LAT_DICT:
            self.instrnMem_lat = self.INSTRN_MEM_LAT_DICT[self.instrnMem_size]
            self.instrnMem_pow_dyn = self.INSTRN_MEM_POW_DYN_DICT[self.instrnMem_size]
            self.instrnMem_pow_leak = self.INSTRN_MEM_POW_LEAK_DICT[self.instrnMem_size]
            self.instrnMem_area = self.INSTRN_MEM_AREA_DICT[self.instrnMem_size] * math.sqrt(8)  # Aligned with PUMA

        return self
