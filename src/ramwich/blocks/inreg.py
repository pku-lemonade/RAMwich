import numpy as np
from numpy.typing import NDArray

from ..config import MVMUConfig, XBARConfig
from ..stats import Stats

class InRegisterArray:
    """Hardware implementation of the MVMU input register component
    
    MVMU input register array reads differently from normal registers.
    For each read, it reads a several bits and then shifts the register"""

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

    def read(self, bits: int):
        """Read the LSBs from register array and then shift the registers to the right by bits"""
        # Read the LSBs from the registers
        lsb = self.registers & ((1 << bits) - 1)

        # Shift the registers to the right by bits
        self.registers >>= bits

        return lsb
    
    def reset(self):
        """Reset the register array to zero"""
        self.registers = np.zeros(self.size, dtype=np.int32)
    
    def read_all(self):
        """Read all the registers
        This a hack function used for debugging purposes"""
        return self.registers