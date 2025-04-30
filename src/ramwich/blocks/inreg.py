from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..config import MVMUConfig
from .memory import MemoryStats


class InputRegisterArray:
    """Hardware implementation of the MVMU input register component

    MVMU input register array reads differently from normal registers.
    For each read, it reads a several bits and then shifts the register"""

    def __init__(self, mvmu_config: MVMUConfig):
        self.mvmu_config = mvmu_config
        self.size = self.mvmu_config.xbar_config.xbar_size

        # Initialize the registers
        self.registers = np.zeros(self.size, dtype=np.int32)

        # Initialize stats
        self.stats = MemoryStats(config=self.mvmu_config, memory_type="Input Register Array")

    def write(self, value: Union[NDArray[np.int32], int], start: int = 0):
        """Write values to register array"""

        # Convert to numpy array if it's a single value
        if isinstance(value, int):
            value = np.array([value], dtype=np.int32)

        length = len(value)
        end = start + length

        # Validate input
        if start < 0 or end > self.size:
            raise IndexError(f"Write operation out of range ({start}, {len(value)})")

        # Write the value to the registers
        self.registers[start:end] = value

        # Update stats
        self.stats.write_operations += 1
        self.stats.write_cells += length
        self.stats.total_operations += 1
        self.stats.total_operated_cells += length

    def read(self, bits: int):
        """Read the LSBs from register array and then shift the registers to the right by bits"""
        # Read the LSBs from the registers
        lsb = self.registers & ((1 << bits) - 1)

        # Shift the registers to the right by bits
        self.registers >>= bits

        # Update stats
        self.stats.read_operations += 1
        self.stats.read_cells += self.size
        self.stats.total_operations += 1
        self.stats.total_operated_cells += self.size

        return lsb

    def reset(self):
        """Reset the register array to zero"""
        self.registers.fill(0)

        # Update stats
        self.stats.write_operations += 1
        self.stats.write_cells += self.size
        self.stats.total_operations += 1
        self.stats.total_operated_cells += self.size

    def read_all(self):
        """Read all the registers
        This a hack function used for debugging purposes"""
        return self.registers

    def get_stats(self):
        return self.stats.get_stats()
