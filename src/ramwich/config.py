import math
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class BitConfig(str, Enum):
    SLC = "1"
    MLC = "2"
    TLC = "3"
    QLC = "4"
    SRAM = "s"


class DataConfig(BaseModel):
    """Data type configuration"""

    storage_config: list[BitConfig] = Field(
        default=[BitConfig.MLC, BitConfig.MLC, BitConfig.MLC, BitConfig.MLC], description="Storage configuration"
    )
    int_bits: int = Field(default=8, description="Integer bits")
    frac_bits: int = Field(default=0, description="Fractional bits")
    data_bits: int = Field(default=None, init=False, description="Data bits")

    def __init__(self, **data):
        super().__init__(**data)

        self.data_bits = self.int_bits + self.frac_bits


class DACConfig(BaseModel):
    """Digital-to-Analog Converter configuration"""

    # Class constants for lookup tables
    LAT_DICT: ClassVar[dict[int, int]] = {1: 1, 2: 1, 4: 1, 8: 1, 16: 1}
    POW_DYN_DICT: ClassVar[dict[int, float]] = {
        1: 0.00350625,
        2: 0.00350625,
        4: 0.00350625,
        8: 0.00350625,
        16: 0.00350625,
    }
    POW_LEAK_DICT: ClassVar[dict[int, float]] = {
        1: 0.000390625,
        2: 0.000390625,
        4: 0.000390625,
        8: 0.000390625,
        16: 0.000390625,
    }
    AREA_DICT: ClassVar[dict[int, float]] = {1: 1.67e-7, 2: 1.67e-7, 4: 1.67e-7, 8: 1.67e-7, 16: 1.67e-7}

    resolution: int = Field(default=1, description="DAC resolution")
    VDD: float = Field(default=1, description="Supply voltage")

    lat: float = Field(default=None, init=False, description="DAC latency")
    pow_dyn: float = Field(default=None, init=False, description="DAC dynamic power")
    pow_leak: float = Field(default=None, init=False, description="DAC leakage power")
    area: float = Field(default=None, init=False, description="DAC area")

    def __init__(self, **data):
        super().__init__(**data)
        # Update derived values if resolution is different from default
        if self.resolution in self.LAT_DICT:
            self.lat = self.LAT_DICT[self.resolution]
            self.pow_dyn = self.POW_DYN_DICT[self.resolution]
            self.pow_leak = self.POW_LEAK_DICT[self.resolution]
            self.area = self.AREA_DICT[self.resolution]


class XBARConfig(BaseModel):
    """Crossbar and its IO register configuration"""

    # XBAR in memory lookup tables
    INMEM_LAT_DICT: ClassVar[dict[int, int]] = {32: 1, 64: 1, 128: 1, 256: 1}
    INMEM_POW_DYN_READ_DICT: ClassVar[dict[int, int]] = {32: 0.3, 64: 0.7, 128: 1.7, 256: 4.7}
    INMEM_POW_DYN_WRITE_DICT: ClassVar[dict[int, int]] = {32: 0.1, 64: 0.1, 128: 0.16, 256: 0.2}
    INMEM_POW_LEAK_DICT: ClassVar[dict[int, int]] = {32: 0.009, 64: 0.02, 128: 0.04, 256: 0.075}
    INMEM_AREA_DICT: ClassVar[dict[int, int]] = {32: 0.00015, 64: 0.00033, 128: 0.00078, 256: 0.0019}

    inMem_lat: float = Field(default=None, init=False, description="Crossbar input memory latency")
    inMem_pow_dyn_read: float = Field(default=None, init=False, description="Crossbar input memory dynamic read power")
    inMem_pow_dyn_write: float = Field(
        default=None, init=False, description="Crossbar input memory dynamic write power"
    )
    inMem_pow_leak: float = Field(default=None, init=False, description="Crossbar input memory leakage power")
    inMem_area: float = Field(default=None, init=False, description="Crossbar input memory area")

    # XBAR lookup tables
    XBAR_LAT_DICT: ClassVar[dict[int, int]] = {32: 32, 64: 64, 128: 128, 256: 256}
    XBAR_POW_DICT: ClassVar[dict[int, int]] = {32: 0.01875, 64: 0.075, 128: 0.3, 256: 1.2}
    XBAR_AREA_DICT: ClassVar[dict[int, int]] = {32: 1.5625e-6, 64: 6.25e-6, 128: 2.5e-5, 256: 1.0e-4}

    xbar_lat: float = Field(default=None, init=False, description="Crossbar latency")
    xbar_pow: float = Field(default=None, init=False, description="Crossbar power")
    xbar_pow_leak: float = Field(default=0, description="Crossbar leakage power")
    xbar_area: float = Field(default=None, init=False, description="Crossbar area")

    sram_xbar_lat: float = Field(default=None, init=False, description="Crossbar latency")
    sram_xbar_pow: float = Field(default=None, init=False, description="Crossbar power")
    sram_xbar_pow_leak: float = Field(default=0, description="Crossbar leakage power")
    sram_xbar_area: float = Field(default=None, init=False, description="Crossbar area")

    calculator_lat: float = Field(default=1, description="Single SRAM CIM calculator processing latency")
    calculator_pow_leak: float = Field(default=0, description="Single SRAM CIM calculator leakage power")
    calculator_pow_dyn: float = Field(default=0, description="Single SRAM CIM calculator dynamic power")
    calculator_area: float = Field(default=0, description="Single SRAM CIM calculator area")

    # XBAR out memory lookup tables
    OUTMEM_LAT_DICT: ClassVar[dict[int, int]] = {32: 1, 64: 1, 128: 1, 256: 1}
    OUTMEM_POW_DYN_DICT: ClassVar[dict[int, int]] = {32: 0.1, 64: 0.1, 128: 0.16, 256: 0.2}
    OUTMEM_POW_LEAK_DICT: ClassVar[dict[int, int]] = {32: 0.009, 64: 0.02, 128: 0.04, 256: 0.075}
    OUTMEM_AREA_DICT: ClassVar[dict[int, int]] = {32: 0.00015, 64: 0.00033, 128: 0.00078, 256: 0.0019}

    outMem_lat: float = Field(default=None, init=False, description="Crossbar output memory latency")
    outMem_pow_dyn: float = Field(default=None, init=False, description="Crossbar output memory dynamic write power")
    outMem_pow_leak: float = Field(default=None, init=False, description="Crossbar output memory leakage power")
    outMem_area: float = Field(default=None, init=False, description="Crossbar output memory area")

    # Set default values for derived fields instead of None
    xbar_ip_lat: float = Field(default=100.0, description="XBAR input processing latency")
    xbar_ip_pow: float = Field(default=1.37 * 2.0 - 1.04, description="XBAR input processing power")
    xbar_op_lat: float = Field(default=20.0 * 12.8, description="XBAR output processing latency")
    xbar_op_pow: float = Field(default=4.44 * 3.27 / 12.8, description="XBAR output processing power")
    xbar_rd_lat: float = Field(default=328.0 * 1000 * (1 / 32.0), description="XBAR read latency")
    xbar_wr_lat: float = Field(default=351.0 * 1000 * (1 / 32.0), description="XBAR write latency")
    xbar_rd_pow: float = Field(
        default=208.0 * 1000 * (1 / 32.0) / (328.0 * 1000 * (1 / 32.0)), description="XBAR read power"
    )
    xbar_wr_pow: float = Field(
        default=676.0 * 1000 * (1 / 32.0) / (328.0 * 1000 * (1 / 32.0)), description="XBAR write power"
    )

    rram_conductance_min: float = Field(default=0, description="Min value of RRAM conductance")
    rram_conductance_max: float = Field(default=1, description="Max value of RRAM conductance")

    xbar_size: int = Field(default=128, description="Crossbar size")
    noise_sigma: float = Field(default=0, description="RRAM read and calculate noise sigma")
    has_noise: bool = Field(default=False, description="Whether to add noise to the crossbar")

    def __init__(self, **data):
        super().__init__(**data)

        # Override xbar parameters based on xbar_size if it differs from default
        if self.xbar_size in self.XBAR_LAT_DICT:
            self.xbar_lat = self.XBAR_LAT_DICT[self.xbar_size]
            self.xbar_pow = self.XBAR_POW_DICT[self.xbar_size]
            self.xbar_area = self.XBAR_AREA_DICT[self.xbar_size]
            self.inMem_lat = self.INMEM_LAT_DICT[self.xbar_size]
            self.inMem_pow_dyn_read = self.INMEM_POW_DYN_READ_DICT[self.xbar_size]
            self.inMem_pow_dyn_write = self.INMEM_POW_DYN_WRITE_DICT[self.xbar_size]
            self.inMem_pow_leak = self.INMEM_POW_LEAK_DICT[self.xbar_size]
            self.inMem_area = self.INMEM_AREA_DICT[self.xbar_size]
            self.outMem_lat = self.OUTMEM_LAT_DICT[self.xbar_size]
            self.outMem_pow_dyn = self.OUTMEM_POW_DYN_DICT[self.xbar_size]
            self.outMem_pow_leak = self.OUTMEM_POW_LEAK_DICT[self.xbar_size]
            self.outMem_area = self.OUTMEM_AREA_DICT[self.xbar_size]

            # Match PUMA
            self.xbar_pow = self.xbar_ip_lat * self.xbar_ip_pow


class ADCType(str, Enum):
    NORMAL = "normal"
    DIFFERENTIAL = "differential"


class ADCConfig(BaseModel):
    """Analog-to-Digital Converter configuration"""

    type: ADCType = Field(default=ADCType.NORMAL, description="ADC type")

    # Class constants for lookup tables - using integers as keys instead of strings
    LAT_DICT: ClassVar[dict[int, int]] = {1: 13, 2: 25, 4: 50, 8: 100, 16: 200}
    POW_DYN_DICT: ClassVar[dict[int, float]] = {1: 0.225, 2: 0.45, 4: 0.9, 8: 1.8, 16: 3.2}
    POW_LEAK_DICT: ClassVar[dict[int, float]] = {1: 0.025, 2: 0.05, 4: 0.1, 8: 0.2, 16: 0.4}
    AREA_DICT: ClassVar[dict[int, float]] = {1: 0.0012, 2: 0.0012, 4: 0.0012, 8: 0.0012, 16: 0.0012}

    resolution: int = Field(default=8, description="ADC resolution")

    lat: float = Field(default=None, init=False, description="ADC latency")
    pow_dyn: float = Field(default=None, init=False, description="ADC dynamic power")
    pow_leak: float = Field(default=None, init=False, description="ADC leakage power")
    area: float = Field(default=None, init=False, description="ADC area")

    def __init__(self, **data):
        super().__init__(**data)
        # Update derived values based on resolution if it's different from default
        if self.resolution in self.LAT_DICT:
            self.lat = self.LAT_DICT[self.resolution]
            self.pow_dyn = self.POW_DYN_DICT[self.resolution]
            self.pow_leak = self.POW_LEAK_DICT[self.resolution]
            self.area = self.AREA_DICT[self.resolution]


class NOCConfig(BaseModel):
    """Network-on-Chip configuration"""

    # Class constants for lookup tables
    INJ_RATE_MAX: ClassVar[int] = 0.025
    # Convert string keys to numeric keys for consistency
    LAT_DICT: ClassVar[dict[int, int]] = {0.001: 29, 0.005: 31, 0.01: 34, 0.02: 54, 0.025: 115}
    AREA_DICT: ClassVar[dict[int, float]] = {4: 0.047, 8: 0.116}
    POW_DYN_DICT: ClassVar[dict[int, float]] = {4: 16.13, 8: 51.48}
    POW_LEAK_DICT: ClassVar[dict[int, float]] = {4: 0.41, 8: 1.04}

    inj_rate: float = Field(default=0.005, description="Injection rate")
    num_port: int = Field(default=4, description="Number of ports")

    # Hypertransport network defaults
    noc_ht_lat: float = Field(default=5, description="Hypertransport latency")
    noc_inter_lat: float = Field(default=36, description="NoC inter-node latency")
    noc_inter_pow_dyn: float = Field(default=10400, description="NoC inter-node dynamic power")
    noc_inter_pow_leak: float = Field(default=0, description="NoC inter-node leakage power")
    noc_inter_area: float = Field(default=22.88, description="NoC inter-node area")

    # Intra-node network defaults
    noc_intra_lat: float = Field(default=None, init=False, description="NoC intra-node latency")
    noc_intra_pow_dyn: float = Field(default=None, init=False, description="NoC intra-node dynamic power")
    noc_intra_pow_leak: float = Field(default=None, init=False, description="NoC intra-node leakage power")
    noc_intra_area: float = Field(default=None, init=False, description="NoC intra-node area")

    def __init__(self, **data):
        super().__init__(**data)
        # Validate injection rate
        if self.inj_rate > self.INJ_RATE_MAX:
            raise ValueError("NoC injection rate too high! Reconsider NOC design or DNN mapping.")

        # Update derived values based on configuration
        if self.inj_rate in self.LAT_DICT:
            self.noc_intra_lat = self.LAT_DICT[self.inj_rate]

        if self.num_port in self.POW_DYN_DICT:
            self.noc_intra_pow_dyn = self.POW_DYN_DICT[self.num_port]

        if self.num_port in self.POW_LEAK_DICT:
            self.noc_intra_pow_leak = self.POW_LEAK_DICT[self.num_port]

        if self.num_port in self.AREA_DICT:
            self.noc_intra_area = self.AREA_DICT[self.num_port]

        # Update inter-node latency based on intra-node latency
        self.noc_inter_lat = self.noc_ht_lat + self.noc_intra_lat


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

    def __init__(self, **data):
        super().__init__(**data)

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

    dac_config: DACConfig = Field(default_factory=DACConfig)
    xbar_config: XBARConfig = Field(default_factory=XBARConfig)
    adc_config: ADCConfig = Field(default_factory=ADCConfig)

    def __init__(self, **data):
        super().__init__(**data)

        self.num_adc_per_xbar = self.xbar_config.xbar_size // self.num_columns_per_adc

        # Then verify it's a clean division
        if self.xbar_config.xbar_size % self.num_columns_per_adc != 0:
            raise ValueError(
                f"xbar_size ({self.xbar_config.xbar_size}) must be exactly divisible by "
                f"num_columns_per_adc ({self.num_columns_per_adc})"
            )


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)
    """Configuration for the RAMwich Simulator"""
    num_nodes: int = Field(default=1, description="Number of nodes in the system")
    num_tiles_per_node: int = Field(default=4, description="Number of tiles per node")
    num_cores_per_tile: int = Field(default=8, description="Number of cores per tile")
    num_mvmus_per_core: int = Field(default=6, description="Number of MVMUs per core")

    addr_width: int = Field(default=32, description="Address width")
    data_width: int = Field(default=8, description="Data width")
    instrn_width: int = Field(default=48, description="Instruction width")

    # Add configuration for components with default factories
    data_config: DataConfig = Field(default_factory=DataConfig)
    noc_config: NOCConfig = Field(default_factory=NOCConfig)
    tile_config: TileConfig = Field(default_factory=TileConfig)
    core_config: CoreConfig = Field(default_factory=CoreConfig)
    mvmu_config: MVMUConfig = Field(default_factory=MVMUConfig)

    def __init__(self, **data):
        super().__init__(**data)

        self.mvmu_config.stored_bit = []
        self.mvmu_config.bits_per_cell = []
        self.mvmu_config.is_xbar_rram = []
        self.mvmu_config.rram_to_output_map = []
        self.mvmu_config.sram_to_output_map = []

        bits = 0  # total bits number in the operand
        self.mvmu_config.num_rram_xbar_per_mvmu = 0  # number of RRAM xbars
        self.mvmu_config.num_sram_xbar_per_mvmu = 0  # number of SRAM xbars
        for i in self.data_config.storage_config:
            self.mvmu_config.stored_bit.append(bits)
            if i == BitConfig.SRAM:
                self.mvmu_config.num_sram_xbar_per_mvmu += 1
                self.mvmu_config.bits_per_cell.append(1)
                self.mvmu_config.is_xbar_rram.append(False)
                bits += 1
            else:
                self.mvmu_config.num_rram_xbar_per_mvmu += 1
                self.mvmu_config.bits_per_cell.append(int(i))
                self.mvmu_config.is_xbar_rram.append(True)
                bits += int(i)
        self.mvmu_config.stored_bit.append(bits)

        self.mvmu_config.num_xbar_per_mvmu = (
            self.mvmu_config.num_sram_xbar_per_mvmu + self.mvmu_config.num_rram_xbar_per_mvmu
        )

        for i in range(self.mvmu_config.num_xbar_per_mvmu):
            if self.mvmu_config.is_xbar_rram[i]:
                self.mvmu_config.rram_to_output_map.append(i)
            else:
                self.mvmu_config.sram_to_output_map.append(i)

        assert bits == self.data_width, "storage config invalid: check if total bits in storage config = data width"
        assert self.data_config.int_bits + self.data_config.frac_bits == self.data_width, (
            "storage config invalid: check if total bits in storage config = int_bits + frac_bits"
        )

        self.tile_config.edram_size = self.tile_config.edram_size_in_KB * 1024 * 8 // self.data_width
