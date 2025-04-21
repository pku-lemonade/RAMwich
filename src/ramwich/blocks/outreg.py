from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..config import MVMUConfig
from ..stats import Stats
from .memory import MemoryStats


class OutputRegisterArray:
    """Hardware implementation of the MVMU input register component

    MVMU output register array is just arrayed normal registers."""

    def __init__(self, mvmu_config: MVMUConfig = None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.size = self.mvmu_config.xbar_config.xbar_size

        # Initialize the registers
        self.registers = np.zeros(self.size, dtype=np.int32)

        # Initialize stats
        self.stats = MemoryStats()

    def write(self, value: NDArray[np.int32], indices: Optional[NDArray[np.int32]] = None):
        """Write values to specific indices in the register array"""

        if indices is None:
            # If no indices are provided, write the entire value
            # Validate input
            if value.shape != self.registers.shape:
                raise ValueError("Value shape does not match register array shape")

            np.copyto(self.registers, value)

            # Update stats
            self.stats.write_operations += self.size
            self.stats.total_operations += self.size

        else:
            # Validate indices
            if np.any(indices >= self.size) or np.any(indices < 0):
                raise ValueError("Index out of bounds")

            self.registers[indices] = value

            # Update stats
            length = len(indices)
            self.stats.write_operations += length
            self.stats.total_operations += length

    def read(self, indices: Optional[NDArray[np.int32]] = None):
        """Read specific indices from the register array"""
        if indices is None:
            # If no indices are provided, read the entire register array
            # Update stats
            self.stats.read_operations += self.size
            self.stats.total_operations += self.size

            return self.registers.copy()

        # Validate indices
        if np.any(indices >= self.size) or np.any(indices < 0):
            raise ValueError("Index out of bounds")

        # Update stats
        length = len(indices)
        self.stats.read_operations += length
        self.stats.total_operations += length

        return self.registers[indices].copy()
        # Technically, .copy() is not needed.
        # But we keep it for clearence and maintain consistency with the write method

    def reset(self):
        """Reset the register array to zero"""
        self.registers.fill(0)

        # Update stats
        self.stats.write_operations += self.size
        self.stats.total_operations += self.size

    def get_stats(self):
        return self.stats.get_stats()
