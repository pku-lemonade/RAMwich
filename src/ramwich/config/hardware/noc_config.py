from typing import ClassVar
from pydantic import BaseModel, Field

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
