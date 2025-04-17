from typing import Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import Config
from ..stats import Stats


class SRAMStats(BaseModel):
    """Statistics for SRAM operations"""

    read_operations: int = Field(default=0, description="Number of read operations")
    write_operations: int = Field(default=0, description="Number of write operations")
    total_operations: int = Field(default=0, description="Total number of operations")

    def get_stats(self) -> Stats:
        """Convert SRAMStats to general Stats object"""
        stats = Stats()
        stats.latency = 0.0  # Will be updated through execution
        stats.operations = self.total_operations
        stats.read_operations = self.read_operations
        stats.write_operations = self.write_operations
        return stats


class SRAM:
    """SRAM register file component for the Core"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.size = self.config.core_config.dataMem_size
        # Address bias for the SRAM registers
        # Address 0 to num_mvmus_per_core * xbar_size - 1 is for the MVMU input registers
        # Address num_mvmus_per_core * xbar_size to num_mvmus_per_core * xbar_size * 2 - 1 is for the MVMU output registers
        # From num_mvmus_per_core * xbar_size * 2 and above is for the core cache
        self.address_bias = self.config.num_mvmus_per_core * self.config.mvmu_config.xbar_size * 2

        # Initialize registers
        self.registers = np.zeros(self.size, dtype=np.int32)

        # Initialize stats
        self.stats = SRAMStats()

    def read(self, start: int, length: int):
        """Read a block of registers from SRAM"""

        # Validate input
        if start < self.address_bias or start + length > self.address_bias + self.size:
            raise IndexError(f"Read operation out of range ({start}, {length})")
        if length <= 0:
            raise ValueError("Length must be a positive integer")

        # Calculate internal indices
        internal_start = start - self.address_bias
        internal_end = internal_start + length

        # Update stats
        self._update_stats("read")

        return self.registers[internal_start:internal_end].copy()

    def write(self, start: int, values: Union[NDArray[np.integer], int]):
        """Write values to a block of registers in SRAM"""
        # Convert to numpy array if it's a single value
        if isinstance(values, int):
            values = np.array([values], dtype=np.int32)

        length = len(values)

        # Validate input
        if start < self.address_bias or start + length > self.address_bias + self.size:
            raise IndexError(f"Write operation out of range ({start}, {length})")

        # Calculate internal indices
        internal_start = start - self.address_bias
        internal_end = internal_start + length

        # Write values
        self.registers[internal_start:internal_end] = values

        # Update stats
        self._update_stats("write")

    def _update_stats(self, operation_type: str) -> None:
        self.stats.total_operations += 1
        if operation_type == "read":
            self.stats.read_operations += 1
        elif operation_type == "write":
            self.stats.write_operations += 1
        self.stats.latency += self.latency

    def get_stats(self) -> Stats:
        return self.stats.get_stats()
