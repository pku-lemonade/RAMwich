import numpy as np
from numpy.typing import NDArray

from ..config import MVMUConfig
from ..stats import Stats

class OutRegisterArray:
    """Hardware implementation of the MVMU input register component
    
    MVMU output register array is just arrayed normal registers."""

    def __init__(self, mvmu_config: MVMUConfig=None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.size = self.mvmu_config.xbar_config.xbar_size

        # Initialize the registers
        self.registers = np.zeros(self.size, dtype=np.int32)

        # Initialize stats

    def write(self, value: NDArray[np.integer]):
        """Write values to register array"""

        if value.shape != self.registers.shape:
            raise ValueError("Value shape does not match register array shape")
        
        # Write the value to the registers
        self.registers = np.copy(value)

    def read(self):
        """Read the register array"""
        return self.registers
    
    def reset(self):
        """Reset the register array to zero"""
        self.registers = np.zeros(self.size, dtype=np.int32)