from typing import Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import CoreConfig, TileConfig
from ..stats import Stats


class MemoryStats(BaseModel):
    """Statistics for DRAM operations"""

    read_operations: int = Field(default=0, description="Number of read operations")
    write_operations: int = Field(default=0, description="Number of write operations")
    total_operations: int = Field(default=0, description="Total number of operations")

    def get_stats(self) -> Stats:
        """Convert MemoryStats to general Stats object"""
        stats = Stats()
        stats.latency = 0.0  # Will be updated through update_execution_time
        stats.energy = 0.0  # Placeholder for energy consumption
        stats.area = 0.0  # Placeholder for area usage
        stats.operations = self.total_operations
        stats.read_operations = self.read_operations
        stats.write_operations = self.write_operations
        return stats


class Memory:
    """Base memory component"""

    def __init__(self, size: int):
        self.size = size

        # Initialize memory cells
        self.cells = np.zeros(self.size, dtype=np.int32)

        # Initialize stats
        self.stats = MemoryStats()

    def read(self, start: int, length: int):
        """Read a block of registers from SRAM"""

        end = start + length

        # Validate input
        if start < 0 or end > self.size:
            raise IndexError(f"Read operation out of range ({start}, {length})")
        if length <= 0:
            raise ValueError("Length must be a positive integer")

        # Update stats
        self.stats.read_operations += length
        self.stats.total_operations += length

        return self.cells[start:end].copy()

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
        self.cells[start:end] = values

        # Update stats
        self.stats.write_operations += length
        self.stats.total_operations += length

    def get_stats(self) -> Stats:
        return self.stats.get_stats()


class SRAM(Memory):
    """SRAM registers file component for the Core"""

    def __init__(self, core_config: CoreConfig = None):
        self.core_config = core_config or CoreConfig()
        size = self.core_config.dataMem_size
        super().__init__(size)


class DRAM(Memory):
    """DRAM array component for the Tile"""

    def __init__(self, tile_config: TileConfig = None):
        self.tile_config = tile_config or TileConfig()
        size = self.tile_config.edram_size
        super().__init__(size)
