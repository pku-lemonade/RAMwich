import numpy as np
from pydantic import BaseModel, Field

from ..config import MVMUConfig
from ..stats import Stats, StatsDict


class SNHStats(BaseModel):
    """Statistics tracking for Sample-and-Hold (SNH) components"""

    # Universal metrics
    config: MVMUConfig = Field(default=MVMUConfig(), description="SNH configuration")
    size: int = Field(default=0, description="Size of the SNH array")

    # SNH specific metrics
    samples: int = Field(default=0, description="Number of samples taken")

    def get_stats(self) -> StatsDict:
        """Convert SNHStats to general Stats object"""
        stats = Stats(
            activation_count=self.samples,
            dynamic_energy=self.config.snh_pow_dyn * self.samples,
            leakage_energy=self.config.snh_pow_leak * self.size,
            area=self.config.snh_area * self.size,
        )

        return StatsDict({"Sample and Hold": stats})


class SNHArray:
    """
    Hardware implementation of Sample-and-Hold (SNH) array for storing sampled values
    this will do nothing functionally, but it is a inportant part of the hardware.
    Its energy consumption and area will be calculated
    """

    def __init__(self, mvmu_config: MVMUConfig = None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.shape = (self.mvmu_config.num_rram_xbar_per_mvmu, self.mvmu_config.xbar_config.xbar_size)
        self.size = np.prod(self.shape)

        # Initialize stats
        self.stats = SNHStats(config=self.mvmu_config, size=self.size)

    def sample(self):
        """Sample the input data"""
        # Update the stats
        self.stats.samples += self.size

    def get_stats(self) -> StatsDict:
        """Get the statistics for this SNH array"""
        return self.stats.get_stats()
