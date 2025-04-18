from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..config import MVMUConfig
from ..stats import Stats
from .sram import SRAMStats


class OutputRegisterArray:
    """Hardware implementation of the MVMU input register component

    MVMU output register array is just arrayed normal registers."""

    def __init__(self, mvmu_config: MVMUConfig = None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.size = self.mvmu_config.xbar_config.xbar_size

        # Initialize the registers
        self.registers = np.zeros(self.size, dtype=np.int32)

        # Initialize stats
        self.stats = SRAMStats()

    def write(self, value: NDArray[np.int32], indices: Optional[NDArray[np.int32]] = None):
        """Write values to specific indices in the register array"""

        if indices is None:
            # If no indices are provided, write the entire value
            # Validate input
            if value.shape != self.registers.shape:
                raise ValueError("Value shape does not match register array shape")

            self.registers = np.copy(value)

            # Update stats
            self._update_stats("write", self.size)

        else:
            # Validate indices
            if np.any(indices >= self.size) or np.any(indices < 0):
                raise ValueError("Index out of bounds")

            self.registers[indices] = np.copy(value)

            # Update stats
            self._update_stats("write", len(indices))

    def read(self, indices: Optional[NDArray[np.int32]] = None):
        """Read specific indices from the register array"""
        if indices is None:
            # If no indices are provided, read the entire register array
            # Update stats
            self._update_stats("read", self.size)

            return self.registers

        # Validate indices
        if np.any(indices >= self.size) or np.any(indices < 0):
            raise ValueError("Index out of bounds")

        # Update stats
        self._update_stats("read", len(indices))

        return self.registers[indices]

    def read_clipped(self, discard_bits: int, indices: Optional[NDArray[np.int32]] = None):
        """Read specific indices from the register array and discard bits"""
        if indices is None:
            # If no indices are provided, read the entire register array
            # Update stats
            self._update_stats("read", self.size)

            return self.registers >> discard_bits

        # Validate indices
        if np.any(indices >= self.size) or np.any(indices < 0):
            raise ValueError("Index out of bounds")

        # Update stats
        self._update_stats("read", len(indices))

        return (self.registers[indices] >> discard_bits).astype(np.int32)

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
