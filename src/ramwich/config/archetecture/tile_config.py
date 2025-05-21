import math
from typing import ClassVar
from pydantic import BaseModel, Field

class TileConfig(BaseModel):
    """Tile configuration"""

    # Tile Control unit
    tcu_pow_dyn: float = Field(default=0.25 * 0.2, description="Tile control unit dynamic power")
    tcu_pow_leak: float = Field(default=0, description="Tile control unit leakage power")
    tcu_area: float = Field(default=0.00145, description="Tile control unit area")

    # EDRAM lookup tables
    EDRAM_LAT_DICT: ClassVar[dict[int, int]] = {8: 50, 64: 50, 128: 50, 2048: 50, 8192: 50, 16384: 50}
    EDRAM_POW_DYN_DICT: ClassVar[dict[int, float]] = {
        8: 17.2 / 50,
        64: 17.2 / 50,
        128: 25.35 / 50,
        2048: 25.35 / 50,
        8192: 25.35 / 50,
        16384: 25.35 / 50,
    }
    EDRAM_POW_LEAK_DICT: ClassVar[dict[int, float]] = {
        8: 0.46,
        64: 0.46,
        128: 0.77,
        2048: 0.77,
        8192: 0.77,
        16384: 0.77,
    }
    EDRAM_AREA_DICT: ClassVar[dict[int, float]] = {
        8: 0.086,
        64: 0.086,
        128: 0.121,
        2048: 0.121,
        8192: 0.121,
        16384: 0.121,
    }

    edram_size_in_KB: int = Field(default=8192, description="EDRAM size in KB")
    edram_size: int = Field(default=4194304, description="EDRAM size")
    edram_lat: float = Field(default=None, init=False, description="EDRAM latency")
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

    instrnMem_size: int = Field(default=131072, description="Tile instruction memory size")
    instrnMem_lat: float = Field(default=None, init=False, description="Tile instruction memory latency")
    instrnMem_pow_dyn: float = Field(default=None, init=False, description="Tile instruction memory dynamic power")
    instrnMem_pow_leak: float = Field(default=None, init=False, description="Tile instruction memory leakage power")
    instrnMem_area: float = Field(default=None, init=False, description="Tile instruction memory area")

    # EDRAM counter buffer values
    counter_buff_lat: float = Field(default=1 * math.sqrt(8), description="Counter buffer latency")
    counter_buff_pow_dyn: float = Field(default=0.65 / 2 * math.sqrt(8), description="Counter buffer dynamic power")
    counter_buff_pow_leak: float = Field(default=0.33 / 2 * math.sqrt(8), description="Counter buffer leakage power")
    counter_buff_area: float = Field(default=0.0041 * math.sqrt(8), description="Counter buffer area")

    # EDRAM to MVMU bus values
    edram_bus_size: int = Field(default=256, description="EDRAM bus size")
    edram_bus_lat: float = Field(default=1, description="EDRAM bus latency")
    edram_bus_pow_dyn: float = Field(
        default=6 / 2, description="EDRAM bus dynamic power"
    )  # bus width = 384, same as issac (over two cycles)
    edram_bus_pow_leak: float = Field(
        default=1 / 2, description="EDRAM bus leakage power"
    )  # bus width = 384, same as issac
    edram_bus_area: float = Field(default=0.090, description="EDRAM bus area")

    # EDRAM controller values
    edram_ctrl_lat: float = Field(default=1, description="EDRAM controller latency")
    edram_ctrl_pow_dyn: float = Field(default=0.475, description="EDRAM controller dynamic power")
    edram_ctrl_pow_leak: float = Field(default=0.05, description="EDRAM controller leakage power")
    edram_ctrl_area: float = Field(default=0.00145, description="EDRAM controller area")

    # Receive buffer value dictionary
    receive_buffer_lat: float = Field(default=1 * math.sqrt(4), description="Receive buffer latency")
    receive_buffer_pow_dyn: float = Field(default=4.48 * math.sqrt(4), description="Receive buffer dynamic power")
    receive_buffer_pow_leak: float = Field(default=0.09 * math.sqrt(4), description="Receive buffer leakage power")
    receive_buffer_area: float = Field(default=0.0022 * math.sqrt(4), description="Receive buffer area")

    def __init__(self, **data):
        super().__init__(**data)

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
