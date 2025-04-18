from typing import Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import CoreConfig
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

    def __init__(self, core_config: CoreConfig = None):
        self.core_config = core_config or CoreConfig()
        self.size = self.core_config.dataMem_size

        # Initialize registers
        self.registers = np.zeros(self.size, dtype=np.int32)

        # Initialize stats
        self.stats = SRAMStats()

    def read(self, start: int, length: int):
        """Read a block of registers from SRAM"""

        end = start + length

        # Validate input
        if start < 0 or end > self.size:
            raise IndexError(f"Read operation out of range ({start}, {length})")
        if length <= 0:
            raise ValueError("Length must be a positive integer")

        # Update stats
        self._update_stats("read", length)

        return self.registers[start:end].copy()

    def write(self, start: int, values: Union[NDArray[np.int32], int]):
        """Write values to a block of registers in SRAM"""
        # Convert to numpy array if it's a single value
        if isinstance(values, int):
            values = np.array([values], dtype=np.int32)

        length = len(values)
        end = start + length

        # Validate input
        if start < 0 or end > self.size:
            raise IndexError(f"Write operation out of range ({start}, {length})")

        # Write values
        self.registers[start:end] = values.copy()

        # Update stats
        self._update_stats("write", length)

    def _update_stats(self, operation_type: str, length) -> None:
        self.stats.total_operations += length
        if operation_type == "read":
            self.stats.read_operations += length
        elif operation_type == "write":
            self.stats.write_operations += length

    def get_stats(self) -> Stats:
        return self.stats.get_stats()
