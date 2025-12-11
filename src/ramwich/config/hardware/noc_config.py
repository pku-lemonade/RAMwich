from typing import ClassVar

from pydantic import BaseModel, Field, model_validator


class NOCConfig(BaseModel):
    """Network-on-Chip configuration"""

    # Class constants for lookup tables
    INJ_RATE_MAX: ClassVar[int] = 0.025
    # Map injection rates to corresponding latencies
    LAT_DICT: ClassVar[dict[float, int]] = {0.001: 8, 0.005: 9, 0.01: 12, 0.02: 20, 0.025: 30}
    AREA_DICT: ClassVar[dict[int, float]] = {4: 0.0222, 8: 0.0548}
    POW_DYN_DICT: ClassVar[dict[int, float]] = {4: 81.8, 8: 95.1}
    POW_LEAK_DICT: ClassVar[dict[int, float]] = {4: 0.328, 8: 0.832}

    inj_rate: float = Field(default=0.005, description="Injection rate")
    num_port: int = Field(default=4, description="Number of ports")

    # Hypertransport network defaults
    noc_ht_lat: int = Field(default=0, description="Hypertransport latency")
    noc_inter_lat: int = Field(default=0, description="NoC inter-node latency")
    noc_inter_pow_dyn: float = Field(default=0, description="NoC inter-node dynamic power")
    noc_inter_pow_leak: float = Field(default=0, description="NoC inter-node leakage power")
    noc_inter_area: float = Field(default=0, description="NoC inter-node area")

    # Intra-node network defaults
    noc_intra_lat: int = Field(default=None, init=False, description="NoC intra-node latency")
    noc_intra_pow_dyn: float = Field(default=None, init=False, description="NoC intra-node dynamic power")
    noc_intra_pow_leak: float = Field(default=None, init=False, description="NoC intra-node leakage power")
    noc_intra_area: float = Field(default=None, init=False, description="NoC intra-node area")

    @model_validator(mode="after")
    def calculate_derived_values(self):
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
        return self
