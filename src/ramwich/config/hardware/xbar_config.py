from typing import ClassVar

from pydantic import BaseModel, Field, model_validator


class XBARConfig(BaseModel):
    """Crossbar and its IO register configuration"""

    # XBAR in memory lookup tables
    INMEM_LAT_DICT: ClassVar[dict[int, float]] = {32: 0.09149, 64: 0.09447, 128: 0.10647, 256: 0.10647}
    INMEM_POW_DYN_READ_DICT: ClassVar[dict[int, float]] = {32: 0.28, 64: 0.50, 128: 0.92, 256: 0.92}
    INMEM_POW_DYN_WRITE_DICT: ClassVar[dict[int, float]] = {32: 0.30, 64: 0.40, 128: 0.58, 256: 0.58}
    INMEM_POW_LEAK_DICT: ClassVar[dict[int, float]] = {32: 0.035, 64: 0.081, 128: 0.151, 256: 0.151}
    INMEM_AREA_DICT: ClassVar[dict[int, float]] = {32: 0.000888364, 64: 0.00191931, 128: 0.00378474, 256: 0.00378474}

    inMem_lat: float = Field(default=None, init=False, description="Crossbar input memory latency")
    inMem_pow_dyn_read: float = Field(default=None, init=False, description="Crossbar input memory dynamic read power")
    inMem_pow_dyn_write: float = Field(
        default=None, init=False, description="Crossbar input memory dynamic write power"
    )
    inMem_pow_leak: float = Field(default=None, init=False, description="Crossbar input memory leakage power")
    inMem_area: float = Field(default=None, init=False, description="Crossbar input memory area")

    # XBAR lookup tables
    XBAR_LAT_DICT: ClassVar[dict[int, float]] = {32: 0.000641, 64: 0.000641, 128: 0.000641, 256: 0.000641}  # ns
    XBAR_POW_DICT: ClassVar[dict[int, float]] = {32: 2.299463, 64: 4.561193, 128: 8.975108, 256: 12.384009}
    XBAR_AREA_DICT: ClassVar[dict[int, float]] = {32: 0.000095, 64: 0.000190, 128: 0.000381, 256: 0.000761}

    xbar_lat: float = Field(default=None, init=False, description="Crossbar latency")
    xbar_pow: float = Field(default=None, init=False, description="Crossbar power")
    xbar_pow_leak: float = Field(default=0, description="Crossbar leakage power")
    xbar_area: float = Field(default=None, init=False, description="Crossbar area")

    SRAM_XBAR_LAT_DICT: ClassVar[dict[int, float]] = {32: 0.035526, 64: 0.071056, 128: 0.142189, 256: 0.284604}  # ns
    SRAM_XBAR_POW_DYN_DICT: ClassVar[dict[int, float]] = {32: 0.511691, 64: 1.014985, 128: 1.997196, 256: 2.755766}
    SRAM_XBAR_POW_LEAK_DICT: ClassVar[dict[int, float]] = {32: 0.0, 64: 0.000002, 128: 0.000008, 256: 0.000031}
    SRAM_XBAR_AREA_DICT: ClassVar[dict[int, float]] = {32: 0.000555, 64: 0.001110, 128: 0.002220, 256: 0.004441}

    sram_xbar_lat: float = Field(default=None, init=False, description="Crossbar latency")
    sram_xbar_pow_dyn: float = Field(default=None, init=False, description="Crossbar power")
    sram_xbar_pow_leak: float = Field(default=None, init=False, description="Crossbar leakage power")
    sram_xbar_area: float = Field(default=None, init=False, description="Crossbar area")

    CALCULATOR_LAT_DICT: ClassVar[dict[int, float]] = {32: 0.0, 64: 0.0, 128: 0.0, 256: 0.0}  # ns
    CALCULATOR_POW_LEAK_DICT: ClassVar[dict[int, float]] = {32: 0.0, 64: 0.0, 128: 0.0, 256: 0.0}
    CALCULATOR_POW_DYN_DICT: ClassVar[dict[int, float]] = {32: 2.769269, 64: 2.769269, 128: 2.769269, 256: 2.769269}
    CALCULATOR_AREA_DICT: ClassVar[dict[int, float]] = {32: 0.001457, 64: 0.001457, 128: 0.001457, 256: 0.001457}

    calculator_lat: float = Field(default=None, init=False, description="Single SRAM CIM calculator processing latency")
    calculator_pow_leak: float = Field(default=None, init=False, description="Single SRAM CIM calculator leakage power")
    calculator_pow_dyn: float = Field(default=None, init=False, description="Single SRAM CIM calculator dynamic power")
    calculator_area: float = Field(default=None, init=False, description="Single SRAM CIM calculator area")

    # XBAR out memory lookup tables
    OUTMEM_LAT_DICT: ClassVar[dict[int, float]] = {32: 0.09149, 64: 0.09447, 128: 0.10647, 256: 0.10647}
    OUTMEM_POW_DYN_DICT: ClassVar[dict[int, float]] = {32: 0.29, 64: 0.45, 128: 0.75, 256: 0.75}
    OUTMEM_POW_LEAK_DICT: ClassVar[dict[int, float]] = {32: 0.035, 64: 0.081, 128: 0.151, 256: 0.151}
    OUTMEM_AREA_DICT: ClassVar[dict[int, float]] = {32: 0.000888364, 64: 0.00191931, 128: 0.00378474, 256: 0.00378474}

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

    rram_conductance_min: float = Field(default=9.8e-6, description="Min value of RRAM conductance (S)")
    rram_conductance_max: float = Field(default=1.67e-4, description="Max value of RRAM conductance (S)")

    xbar_size: int = Field(default=128, description="Crossbar size")
    noise_sigma: float = Field(default=1e-6, description="RRAM read and calculate noise sigma")
    has_noise: bool = Field(default=False, description="Whether to add noise to the crossbar")

    @model_validator(mode="after")
    def calculate_derived_values(self):
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

        return self
