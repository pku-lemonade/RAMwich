import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import MVMUConfig
from ..stats import Stats


class SNHStats(BaseModel):
    """Statistics tracking for Sample-and-Hold (SNH) components"""

    # Universal metrics
    unit_energy_consumption: float = Field(default=0.0, description="Energy consumption for each sample in pJ")
    leakage_energy_per_cycle: float = Field(default=0.0, description="Leakage energy consumption for 1 cycle in pJ")
    area: float = Field(default=0.0, description="Area in mm^2")

    # SNH specific metrics
    samples: int = Field(default=0, description="Number of samples taken")

    def get_stats(self) -> Stats:
        """Convert SNHStats to general Stats object"""
        stats = Stats()

        # Map SNH metrics to Stat object
        stats.increment_component_activation("SNH", self.samples)
        stats.increment_component_dynamic_energy("SNH", self.unit_energy_consumption * self.samples)
        stats.increment_component_leakage_energy("SNH", self.leakage_energy_per_cycle)
        stats.increment_component_area("SNH", self.area)

        return stats


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
        self.stats = SNHStats()
        self.stats.unit_energy_consumption = self.mvmu_config.snh_pow_dyn
        self.stats.leakage_energy_per_cycle = self.mvmu_config.snh_pow_leak * self.size
        self.stats.area = self.mvmu_config.snh_area * self.size

    def sample(self):
        """Sample the input data"""
        # Update the stats
        self.stats.samples += self.size

    def get_stats(self) -> Stats:
        """Get the statistics for this SNH array"""
        return self.stats.get_stats()
