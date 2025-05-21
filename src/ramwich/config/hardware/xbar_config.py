from typing import ClassVar
from pydantic import BaseModel, Field

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

    SRAM_XBAR_LAT_DICT: ClassVar[dict[int, int]] = {32: 1, 64: 1, 128: 1, 256: 1}
    SRAM_XBAR_POW_DYN_DICT: ClassVar[dict[int, int]] = {32: 0.32, 64: 0.64, 128: 1.28, 256: 2.56}
    SRAM_XBAR_POW_LEAK_DICT: ClassVar[dict[int, int]] = {32: 0.02, 64: 0.08, 128: 0.32, 256: 1.28}
    SRAM_XBAR_AREA_DICT: ClassVar[dict[int, int]] = {32: 0.00012, 64: 0.00049, 128: 0.00196, 256: 0.00784}

    sram_xbar_lat: float = Field(default=None, init=False, description="Crossbar latency")
    sram_xbar_pow_dyn: float = Field(default=None, init=False, description="Crossbar power")
    sram_xbar_pow_leak: float = Field(default=None, init=False, description="Crossbar leakage power")
    sram_xbar_area: float = Field(default=None, init=False, description="Crossbar area")

    CALCULATOR_LAT_DICT: ClassVar[dict[int, int]] = {32: 6, 64: 7, 128: 8, 256: 9}
    CALCULATOR_POW_LEAK_DICT: ClassVar[dict[int, int]] = {32: 0.02, 64: 0.04, 128: 0.08, 256: 0.16}
    CALCULATOR_POW_DYN_DICT: ClassVar[dict[int, int]] = {32: 3.24, 64: 6.98, 128: 14.56, 256: 29.82}
    CALCULATOR_AREA_DICT: ClassVar[dict[int, int]] = {32: 0.000058, 64: 0.000127, 128: 0.000265, 256: 0.000545}

    calculator_lat: float = Field(default=None, init=False, description="Single SRAM CIM calculator processing latency")
    calculator_pow_leak: float = Field(default=None, init=False, description="Single SRAM CIM calculator leakage power")
    calculator_pow_dyn: float = Field(default=None, init=False, description="Single SRAM CIM calculator dynamic power")
    calculator_area: float = Field(default=None, init=False, description="Single SRAM CIM calculator area")

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
    noise_sigma: float = Field(default=0.01, description="RRAM read and calculate noise sigma")
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

        # Override SRAM xbar parameters based on xbar_size if it differs from default
        if self.xbar_size in self.SRAM_XBAR_LAT_DICT:
            self.sram_xbar_lat = self.SRAM_XBAR_LAT_DICT[self.xbar_size]
            self.sram_xbar_pow_dyn = self.SRAM_XBAR_POW_DYN_DICT[self.xbar_size]
            self.sram_xbar_area = self.SRAM_XBAR_AREA_DICT[self.xbar_size]
            self.sram_xbar_pow_leak = self.SRAM_XBAR_POW_LEAK_DICT[self.xbar_size]

        # Override calculator parameters based on xbar_size if it differs from default
        if self.xbar_size in self.CALCULATOR_LAT_DICT:
            self.calculator_lat = self.CALCULATOR_LAT_DICT[self.xbar_size]
            self.calculator_pow_leak = self.CALCULATOR_POW_LEAK_DICT[self.xbar_size]
            self.calculator_pow_dyn = self.CALCULATOR_POW_DYN_DICT[self.xbar_size]
            self.calculator_area = self.CALCULATOR_AREA_DICT[self.xbar_size]
