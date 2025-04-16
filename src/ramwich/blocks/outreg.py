from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..config import MVMUConfig
from ..stats import Stats

class OutputRegisterArray:
    """Hardware implementation of the MVMU input register component
    
    MVMU output register array is just arrayed normal registers."""

    def __init__(self, mvmu_config: MVMUConfig=None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.size = self.mvmu_config.xbar_config.xbar_size

        # Initialize the registers
        self.registers = np.zeros(self.size, dtype=np.int32)

        # Initialize stats
    
    def write(self, value: NDArray[np.integer], indices: Optional[NDArray[np.integer]]=None):
        """Write values to specific indices in the register array"""

        if indices is None:
            # If no indices are provided, write the entire value
            # Validate input
            if value.shape != self.registers.shape:
                raise ValueError("Value shape does not match register array shape")
            
            self.registers = np.copy(value)
        
        else:
            # Validate indices
            if np.any(indices >= self.size) or np.any(indices < 0):
                raise ValueError("Index out of bounds")
        
            self.registers[indices] = np.copy(value)
    
    def read(self, indices: Optional[NDArray[np.integer]]=None):
        """Read specific indices from the register array"""
        if indices is None:
            # If no indices are provided, read the entire register array
            return self.registers
        
        # Validate indices
        if np.any(indices >= self.size) or np.any(indices < 0):
            raise ValueError("Index out of bounds")
        
        return self.registers[indices]
    
    def reset(self):
        """Reset the register array to zero"""
        self.registers = np.zeros(self.size, dtype=np.int32)