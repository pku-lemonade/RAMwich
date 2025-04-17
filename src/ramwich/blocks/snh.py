import numpy as np
from numpy.typing import NDArray

from ..config import MVMUConfig
from ..stats import Stats


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
        self.stats = Stats()

    def sample(self):
        """Sample the input data"""
        pass
