import json
import os
from typing import Dict, Any, List

import yaml
from pydantic import BaseModel, Field

class ADCConfig(BaseModel):
    """Analog-to-Digital Converter configuration"""

    # Class constants for lookup tables
    LAT_DICT = {'1': 1, '2': 1, '4': 1, '8': 1, '16': 1}
    POW_DYN_DICT = {'1': 1.8, '2': 1.8, '4': 1.8, '8': 1.8, '16': 1.8}
    POW_LEAK_DICT = {'1': 0.2, '2': 0.2, '4': 0.2, '8': 0.2, '16': 0.2}
    AREA_DICT = {'1': 0.0012, '2': 0.0012, '4': 0.0012, '8': 0.0012, '16': 0.0012}

    resolution: int = Field(default=8, description="ADC resolution")
    adc_lat: float = None
    adc_pow_dyn: float = None
    adc_pow_leak: float = None
    adc_area: float = None
    res_new: Dict[str, int] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set derived values based on config
        self.adc_lat = self.LAT_DICT.get(str(self.resolution), 1)
        self.adc_pow_dyn = self.POW_DYN_DICT.get(str(self.resolution), 1.8)
        self.adc_pow_leak = self.POW_LEAK_DICT.get(str(self.resolution), 0.2)
        self.adc_area = self.AREA_DICT.get(str(self.resolution), 0.0012)

        # Multi-resolution support
        if self.res_new is None:
            self.res_new = {
                'matrix_adc_0': 8,
                'matrix_adc_1': 8,
                'matrix_adc_2': 8,
                'matrix_adc_3': 8
            }


class DACConfig(BaseModel):
    """Digital-to-Analog Converter configuration"""

    # Class constants for lookup tables
    LAT_DICT = {'1': 1, '2': 1, '4': 1, '8': 1, '16': 1}
    POW_DYN_DICT = {'1': 0.00350625, '2': 0.00350625, '4': 0.00350625,
                    '8': 0.00350625, '16': 0.00350625}
    POW_LEAK_DICT = {'1': 0.000390625, '2': 0.000390625, '4': 0.000390625,
                     '8': 0.000390625, '16': 0.000390625}
    AREA_DICT = {'1': 1.67e-7, '2': 1.67e-7, '4': 1.67e-7, '8': 1.67e-7, '16': 1.67e-7}

    resolution: int = Field(default=1, description="DAC resolution")
    dac_lat: float = None
    dac_pow_dyn: float = None
    dac_pow_leak: float = None
    dac_area: float = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set derived values based on config
        self.dac_lat = self.LAT_DICT.get(str(self.resolution), 1)
        self.dac_pow_dyn = self.POW_DYN_DICT.get(str(self.resolution), 0.00350625)
        self.dac_pow_leak = self.POW_LEAK_DICT.get(str(self.resolution), 0.000390625)
        self.dac_area = self.AREA_DICT.get(str(self.resolution), 1.67e-7)


class NOCConfig(BaseModel):
    """Network-on-Chip configuration"""

    # Class constants for lookup tables
    INJ_RATE_MAX = 0.025
    LAT_DICT = {'0.001': 29, '0.005': 31, '0.01': 34, '0.02': 54, '0.025': 115}
    AREA_DICT = {'4': 0.047, '8': 0.116}
    POW_DYN_DICT = {'4': 16.13, '8': 51.48}
    POW_LEAK_DICT = {'4': 0.41, '8': 1.04}

    inj_rate: float = Field(default=0.005, description="Injection rate")
    num_port: int = Field(default=4, description="Number of ports")
    noc_intra_lat: float = None
    noc_intra_pow_dyn: float = None
    noc_intra_pow_leak: float = None
    noc_intra_area: float = None
    noc_ht_lat: float = None
    noc_inter_lat: float = None
    noc_inter_pow_dyn: float = None
    noc_inter_pow_leak: float = None
    noc_inter_area: float = None

    def __init__(self, **data):
        super().__init__(**data)
        # Validate injection rate
        if self.inj_rate > self.INJ_RATE_MAX:
            raise ValueError('NoC injection rate too high! Reconsider NOC design or DNN mapping.')

        # Set derived values
        self.noc_intra_lat = self.LAT_DICT.get(str(self.inj_rate), 31)
        self.noc_intra_pow_dyn = self.POW_DYN_DICT.get(str(self.num_port), 16.13)
        self.noc_intra_pow_leak = self.POW_LEAK_DICT.get(str(self.num_port), 0.41)
        self.noc_intra_area = self.AREA_DICT.get(str(self.num_port), 0.047)

        # Hypertransport network
        self.noc_ht_lat = 5
        self.noc_inter_lat = self.noc_ht_lat + self.noc_intra_lat
        self.noc_inter_pow_dyn = 10400  # 10.4W
        self.noc_inter_pow_leak = 0
        self.noc_inter_area = 22.88


class IMAConfig(BaseModel):
    """In-Memory Accelerator configuration"""

    # XBAR lookup tables
    XBAR_LAT_DICT = {
        '2': {'32': 32, '64': 64, '128': 128, '256': 256},
        '4': {'32': 32, '64': 64, '128': 128, '256': 256},
        '6': {'32': 32, '64': 64, '128': 128, '256': 256}
    }

    XBAR_POW_DICT = {
        '2': {'32': 0.01875, '64': 0.075, '128': 0.3, '256': 1.2},
        '4': {'32': 0.01875, '64': 0.075, '128': 0.3, '256': 1.2},
        '6': {'32': 0.01875, '64': 0.075, '128': 0.3, '256': 1.2}
    }

    XBAR_AREA_DICT = {
        '2': {'32': 1.5625e-6, '64': 6.25e-6, '128': 2.5e-5, '256': 1.0e-4},
        '4': {'32': 1.5625e-6, '64': 6.25e-6, '128': 2.5e-5, '256': 1.0e-4},
        '6': {'32': 1.5625e-6, '64': 6.25e-6, '128': 2.5e-5, '256': 1.0e-4}
    }

    # Operation latency and power constants
    XBAR_IP_LAT = 100.0
    XBAR_IP_POW = 1.37 * 2.0
    XBAR_OP_LAT = 20.0 * 12.8
    XBAR_OP_POW = 4.44 * 3.27 / 12.8
    XBAR_RD_LAT = 328.0 * 1000 * (1/32.0)
    XBAR_WR_LAT = 351.0 * 1000 * (1/32.0)

    xbar_size: int = Field(default=128, description="Crossbar size")
    dataMem_size: int = Field(default=4096, description="Data memory size")
    instrnMem_size: int = Field(default=131072, description="Instruction memory size")
    xbar_bits: int = Field(default=4, description="Crossbar bits")
    ima_xbar_ip_lat: float = None
    ima_xbar_ip_pow: float = None
    ima_xbar_op_lat: float = None
    ima_xbar_op_pow: float = None
    ima_xbar_rd_lat: float = None
    ima_xbar_wr_lat: float = None
    ima_xbar_rd_pow: float = None
    ima_xbar_wr_pow: float = None
    ima_alu_lat: int = None
    ima_alu_pow_dyn: float = None
    ima_alu_pow_leak: float = None
    ima_alu_area: float = None
    ima_dataMem_lat: int = None
    ima_dataMem_pow_dyn: float = None
    ima_dataMem_pow_leak: float = None
    ima_dataMem_area: float = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set derived operation parameters
        self.ima_xbar_ip_lat = self.XBAR_IP_LAT
        self.ima_xbar_ip_pow = self.XBAR_IP_POW
        self.ima_xbar_op_lat = self.XBAR_OP_LAT
        self.ima_xbar_op_pow = self.XBAR_OP_POW
        self.ima_xbar_rd_lat = self.XBAR_RD_LAT
        self.ima_xbar_wr_lat = self.XBAR_WR_LAT
        self.ima_xbar_rd_pow = 208.0 * 1000 * (1/32.0) / self.ima_xbar_rd_lat
        self.ima_xbar_wr_pow = 676.0 * 1000 * (1/32.0) / self.ima_xbar_rd_lat

        # ALU parameters
        self.ima_alu_lat = 1
        self.ima_alu_pow_dyn = 2.4 * 32/45
        self.ima_alu_pow_leak = 0.27 * 32/45
        self.ima_alu_area = 0.00567 * 32/45

        # Memory lookup tables
        DATA_MEM_LAT_DICT = {'256': 1, '512': 1, '1024': 1, '2048': 1}
        DATA_MEM_POW_DYN_DICT = {'256': 0.16, '512': 0.24, '1024': 0.33, '2048': 0.57}
        DATA_MEM_POW_LEAK_DICT = {'256': 0.044, '512': 0.078, '1024': 0.147, '2048': 0.33}
        DATA_MEM_AREA_DICT = {'256': 0.00056, '512': 0.00108, '1024': 0.00192, '2048': 0.00392}

        # Set memory parameters
        self.ima_dataMem_lat = DATA_MEM_LAT_DICT.get(str(self.dataMem_size), 1)
        self.ima_dataMem_pow_dyn = DATA_MEM_POW_DYN_DICT.get(str(self.dataMem_size), 0.33)
        self.ima_dataMem_pow_leak = DATA_MEM_POW_LEAK_DICT.get(str(self.dataMem_size), 0.147)
        self.ima_dataMem_area = DATA_MEM_AREA_DICT.get(str(self.dataMem_size), 0.00192)


class Config(BaseModel):
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

    # Add configuration for components
    adc_config: ADCConfig = None
    dac_config: DACConfig = None
    noc_config: NOCConfig = None
    ima_config: IMAConfig = None

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize component configurations with defaults if not provided
        if self.adc_config is None:
            self.adc_config = ADCConfig()
        if self.dac_config is None:
            self.dac_config = DACConfig()
        if self.noc_config is None:
            self.noc_config = NOCConfig()
        if self.ima_config is None:
            self.ima_config = IMAConfig(xbar_size=self.num_xbars_per_ima)
