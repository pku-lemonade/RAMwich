from typing import ClassVar, Dict

from pydantic import BaseModel, ConfigDict, Field


class ADCConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    """Analog-to-Digital Converter configuration"""

    # Class constants for lookup tables - using integers as keys instead of strings
    LAT_DICT: ClassVar[Dict[int, int]] = {1: 1, 2: 1, 4: 1, 8: 1, 16: 1}
    POW_DYN_DICT: ClassVar[Dict[int, float]] = {1: 1.8, 2: 1.8, 4: 1.8, 8: 1.8, 16: 1.8}
    POW_LEAK_DICT: ClassVar[Dict[int, float]] = {1: 0.2, 2: 0.2, 4: 0.2, 8: 0.2, 16: 0.2}
    AREA_DICT: ClassVar[Dict[int, float]] = {1: 0.0012, 2: 0.0012, 4: 0.0012, 8: 0.0012, 16: 0.0012}

    resolution: int = Field(default=8, description="ADC resolution")
    adc_lat: float = Field(default=1, description="ADC latency")
    adc_pow_dyn: float = Field(default=1.8, description="ADC dynamic power")
    adc_pow_leak: float = Field(default=0.2, description="ADC leakage power")
    adc_area: float = Field(default=0.0012, description="ADC area")
    res_new: Dict[str, int] = Field(
        default={"matrix_adc_0": 8, "matrix_adc_1": 8, "matrix_adc_2": 8, "matrix_adc_3": 8},
        description="Multi-resolution support",
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Update derived values based on resolution if it's different from default
        if self.resolution in self.LAT_DICT:
            self.adc_lat = self.LAT_DICT[self.resolution]
            self.adc_pow_dyn = self.POW_DYN_DICT[self.resolution]
            self.adc_pow_leak = self.POW_LEAK_DICT[self.resolution]
            self.adc_area = self.AREA_DICT[self.resolution]


class DACConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    """Digital-to-Analog Converter configuration"""

    # Class constants for lookup tables
    LAT_DICT: ClassVar[Dict[int, int]] = {1: 1, 2: 1, 4: 1, 8: 1, 16: 1}
    POW_DYN_DICT: ClassVar[Dict[int, float]] = {
        1: 0.00350625,
        2: 0.00350625,
        4: 0.00350625,
        8: 0.00350625,
        16: 0.00350625,
    }
    POW_LEAK_DICT: ClassVar[Dict[int, float]] = {
        1: 0.000390625,
        2: 0.000390625,
        4: 0.000390625,
        8: 0.000390625,
        16: 0.000390625,
    }
    AREA_DICT: ClassVar[Dict[int, float]] = {1: 1.67e-7, 2: 1.67e-7, 4: 1.67e-7, 8: 1.67e-7, 16: 1.67e-7}

    resolution: int = Field(default=1, description="DAC resolution")
    dac_lat: float = Field(default=1, description="DAC latency")
    dac_pow_dyn: float = Field(default=0.00350625, description="DAC dynamic power")
    dac_pow_leak: float = Field(default=0.000390625, description="DAC leakage power")
    dac_area: float = Field(default=1.67e-7, description="DAC area")

    def __init__(self, **data):
        super().__init__(**data)
        # Update derived values if resolution is different from default
        if self.resolution in self.LAT_DICT:
            self.dac_lat = self.LAT_DICT[self.resolution]
            self.dac_pow_dyn = self.POW_DYN_DICT[self.resolution]
            self.dac_pow_leak = self.POW_LEAK_DICT[self.resolution]
            self.dac_area = self.AREA_DICT[self.resolution]


class NOCConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    """Network-on-Chip configuration"""

    # Class constants for lookup tables
    INJ_RATE_MAX: ClassVar[int] = 0.025
    # Convert string keys to numeric keys for consistency
    LAT_DICT: ClassVar[Dict[int, int]] = {0.001: 29, 0.005: 31, 0.01: 34, 0.02: 54, 0.025: 115}
    AREA_DICT: ClassVar[Dict[int, float]] = {4: 0.047, 8: 0.116}
    POW_DYN_DICT: ClassVar[Dict[int, float]] = {4: 16.13, 8: 51.48}
    POW_LEAK_DICT: ClassVar[Dict[int, float]] = {4: 0.41, 8: 1.04}

    inj_rate: float = Field(default=0.005, description="Injection rate")
    num_port: int = Field(default=4, description="Number of ports")

    # Set default values for derived fields instead of None
    noc_intra_lat: float = Field(default=31, description="NoC intra-node latency")
    noc_intra_pow_dyn: float = Field(default=16.13, description="NoC intra-node dynamic power")
    noc_intra_pow_leak: float = Field(default=0.41, description="NoC intra-node leakage power")
    noc_intra_area: float = Field(default=0.047, description="NoC intra-node area")

    # Hypertransport network defaults
    noc_ht_lat: float = Field(default=5, description="Hypertransport latency")
    noc_inter_lat: float = Field(default=36, description="NoC inter-node latency")
    noc_inter_pow_dyn: float = Field(default=10400, description="NoC inter-node dynamic power")
    noc_inter_pow_leak: float = Field(default=0, description="NoC inter-node leakage power")
    noc_inter_area: float = Field(default=22.88, description="NoC inter-node area")

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


class IMAConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    """In-Memory Accelerator configuration"""

    # XBAR lookup tables with numeric keys instead of strings
    XBAR_LAT_DICT: ClassVar[Dict[int, int]] = {
        2: {32: 32, 64: 64, 128: 128, 256: 256},
        4: {32: 32, 64: 64, 128: 128, 256: 256},
        6: {32: 32, 64: 64, 128: 128, 256: 256},
    }

    XBAR_POW_DICT: ClassVar[Dict[int, float]] = {
        2: {32: 0.01875, 64: 0.075, 128: 0.3, 256: 1.2},
        4: {32: 0.01875, 64: 0.075, 128: 0.3, 256: 1.2},
        6: {32: 0.01875, 64: 0.075, 128: 0.3, 256: 1.2},
    }

    XBAR_AREA_DICT: ClassVar[Dict[int, float]] = {
        2: {32: 1.5625e-6, 64: 6.25e-6, 128: 2.5e-5, 256: 1.0e-4},
        4: {32: 1.5625e-6, 64: 6.25e-6, 128: 2.5e-5, 256: 1.0e-4},
        6: {32: 1.5625e-6, 64: 6.25e-6, 128: 2.5e-5, 256: 1.0e-4},
    }

    # Memory lookup tables with numeric keys
    DATA_MEM_LAT_DICT: ClassVar[Dict[int, int]] = {256: 1, 512: 1, 1024: 1, 2048: 1}
    DATA_MEM_POW_DYN_DICT: ClassVar[Dict[int, float]] = {256: 0.16, 512: 0.24, 1024: 0.33, 2048: 0.57}
    DATA_MEM_POW_LEAK_DICT: ClassVar[Dict[int, float]] = {256: 0.044, 512: 0.078, 1024: 0.147, 2048: 0.33}
    DATA_MEM_AREA_DICT: ClassVar[Dict[int, float]] = {256: 0.00056, 512: 0.00108, 1024: 0.00192, 2048: 0.00392}

    xbar_size: int = Field(default=128, description="Crossbar size")
    dataMem_size: int = Field(default=4096, description="Data memory size")
    instrnMem_size: int = Field(default=131072, description="Instruction memory size")
    xbar_bits: int = Field(default=4, description="Crossbar bits")

    # Set default values for derived fields instead of None
    ima_xbar_ip_lat: float = Field(default=100.0, description="XBAR input processing latency")
    ima_xbar_ip_pow: float = Field(default=1.37 * 2.0, description="XBAR input processing power")
    ima_xbar_op_lat: float = Field(default=20.0 * 12.8, description="XBAR output processing latency")
    ima_xbar_op_pow: float = Field(default=4.44 * 3.27 / 12.8, description="XBAR output processing power")
    ima_xbar_rd_lat: float = Field(default=328.0 * 1000 * (1 / 32.0), description="XBAR read latency")
    ima_xbar_wr_lat: float = Field(default=351.0 * 1000 * (1 / 32.0), description="XBAR write latency")
    ima_xbar_rd_pow: float = Field(
        default=208.0 * 1000 * (1 / 32.0) / (328.0 * 1000 * (1 / 32.0)), description="XBAR read power"
    )
    ima_xbar_wr_pow: float = Field(
        default=676.0 * 1000 * (1 / 32.0) / (328.0 * 1000 * (1 / 32.0)), description="XBAR write power"
    )

    # ALU parameters with default fields
    ima_alu_lat: int = Field(default=1, description="ALU latency")
    ima_alu_pow_dyn: float = Field(default=2.4 * 32 / 45, description="ALU dynamic power")
    ima_alu_pow_leak: float = Field(default=0.27 * 32 / 45, description="ALU leakage power")
    ima_alu_area: float = Field(default=0.00567 * 32 / 45, description="ALU area")

    # Memory parameters - these will be overridden based on dataMem_size
    ima_dataMem_lat: int = Field(default=1, description="Data memory latency")
    ima_dataMem_pow_dyn: float = Field(default=0.33, description="Data memory dynamic power")
    ima_dataMem_pow_leak: float = Field(default=0.147, description="Data memory leakage power")
    ima_dataMem_area: float = Field(default=0.00192, description="Data memory area")

    def __init__(self, **data):
        super().__init__(**data)

        # Override dataMem parameters based on dataMem_size if it differs from default
        if self.dataMem_size in self.DATA_MEM_LAT_DICT:
            self.ima_dataMem_lat = self.DATA_MEM_LAT_DICT[self.dataMem_size]
            self.ima_dataMem_pow_dyn = self.DATA_MEM_POW_DYN_DICT[self.dataMem_size]
            self.ima_dataMem_pow_leak = self.DATA_MEM_POW_LEAK_DICT[self.dataMem_size]
            self.ima_dataMem_area = self.DATA_MEM_AREA_DICT[self.dataMem_size]


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)
    """Configuration for the RAMwich Simulator"""
    num_nodes: int = Field(default=1, description="Number of nodes in the system")
    num_tiles_per_node: int = Field(default=4, description="Number of tiles per node")
    num_cores_per_tile: int = Field(default=2, description="Number of cores per tile")
    num_imas_per_core: int = Field(default=2, description="Number of IMAs per core")
    num_xbars_per_ima: int = Field(default=4, description="Number of crossbars per IMA")
    alu_execution_time: int = Field(default=2, description="Execution time for ALU operations")
    load_execution_time: int = Field(default=5, description="Execution time for Load operations")
    set_execution_time: int = Field(default=1, description="Execution time for Set operations")
    mvm_execution_time: int = Field(default=10, description="Execution time for MVM operations")
    simulation_time: int = Field(default=1000, description="Maximum simulation time")

    # Add configuration for components with default factories
    adc_config: ADCConfig = Field(default_factory=ADCConfig)
    dac_config: DACConfig = Field(default_factory=DACConfig)
    noc_config: NOCConfig = Field(default_factory=NOCConfig)
    ima_config: IMAConfig = Field(default_factory=IMAConfig)
