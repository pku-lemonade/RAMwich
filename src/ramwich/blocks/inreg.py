from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..config import MVMUConfig
from ..stats import Stats


class InputRegisterArray:
    """Hardware implementation of the MVMU input register component

    MVMU input register array reads differently from normal registers.
    For each read, it reads a several bits and then shifts the register"""

    def __init__(self, mvmu_config: MVMUConfig = None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.size = self.mvmu_config.xbar_config.xbar_size

        # Initialize the registers
        self.registers = np.zeros(self.size, dtype=np.int32)

        # Initialize stats

    def write(self, value: Union[NDArray[np.integer], int], start: int = 0):
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
        self.registers[start:end] = value.copy()

        # Update stats
        self._update_stats("read", length)

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

        # Update stats
        self._update_stats("write", self.size)

    def _update_stats(self, operation_type: str, length) -> None:
        self.stats.total_operations += length
        if operation_type == "read":
            self.stats.read_operations += length
        elif operation_type == "write":
            self.stats.write_operations += length

    def read_all(self):
        """Read all the registers
        This a hack function used for debugging purposes"""
        return self.registers
