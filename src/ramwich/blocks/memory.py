from typing import Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import CoreConfig, MVMUConfig, TileConfig
from ..stats import Stats, StatsDict


class MemoryStats(BaseModel):
    """Statistics for Memory operations"""

    # Universal metrics
    config: Union[MVMUConfig, CoreConfig, TileConfig] = Field(default=None, description="Configuration object")

    # Memory specific metrics
    memory_type: str = Field(default="", description="Type of memory (SRAM/DRAM)")
    read_operations: int = Field(default=0, description="Number of read operations")
    read_cells: int = Field(default=0, description="Number of read cells")
    write_operations: int = Field(default=0, description="Number of write operations")
    write_cells: int = Field(default=0, description="Number of written cells")
    total_operations: int = Field(default=0, description="Total number of operations")
    total_operated_cells: int = Field(default=0, description="Total number of operated cells")

    def get_stats(self) -> StatsDict:
        """Convert MemoryStats to general Stats object"""

        # Map Memory metrics to StatsDict object
        if self.memory_type == "SRAM":
            stats = Stats(
                activation_count=self.total_operated_cells,
                dynamic_energy=self.config.dataMem_pow_dyn * self.total_operated_cells,
                leakage_energy=self.config.dataMem_pow_leak,
                area=self.config.dataMem_area,
            )
            return StatsDict({"SRAM": stats})

        elif self.memory_type == "DRAM":
            stats = Stats(
                activation_count=self.total_operations,
                dynamic_energy=self.config.edram_pow_dyn * self.total_operations,
                leakage_energy=self.config.edram_pow_leak,
                area=self.config.edram_area,
            )
            return StatsDict({"DRAM": stats})

        elif self.memory_type == "Input Register Array":
            stats = Stats(
                activation_count=self.total_operations,
                dynamic_energy=self.config.xbar_config.inMem_pow_dyn_read * self.read_operations
                + self.config.xbar_config.inMem_pow_dyn_write * self.write_cells,
                leakage_energy=self.config.xbar_config.inMem_pow_leak,
                area=self.config.xbar_config.inMem_area,
            )
            return StatsDict({"Input Register Array": stats})

        elif self.memory_type == "Output Register Array":
            stats = Stats(
                activation_count=self.total_operations,
                dynamic_energy=self.config.xbar_config.outMem_pow_dyn * self.total_operated_cells,
                leakage_energy=self.config.xbar_config.outMem_pow_leak,
                area=self.config.xbar_config.outMem_area,
            )
            return StatsDict({"Output Register Array": stats})

        else:
            raise ValueError(f"Unknown memory type: {self.memory_type}")


class Memory:
    """Base memory component"""

    def __init__(self, size: int):
        self.size = size

        # Initialize memory cells
        self.cells = np.zeros(self.size, dtype=np.int32)

        # Initialize stats
        self.stats = MemoryStats()

    def read(self, start: int, length: int, batch: int = 1):
        """Read a block of registers from SRAM"""

        end = start + length

        # Validate input
        if start < 0 or end > self.size:
            raise IndexError(f"Read operation out of range ({start}, {length})")
        if length <= 0:
            raise ValueError("Length must be a positive integer")

        # Update stats
        self.stats.read_operations += batch
        self.stats.read_cells += length
        self.stats.total_operations += batch
        self.stats.total_operated_cells += length

        return self.cells[start:end].copy()

    def write(self, start: int, values: Union[NDArray[np.int32], int], batch: int = 1):
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
        self.stats.write_operations += batch
        self.stats.write_cells += length
        self.stats.total_operations += batch
        self.stats.total_operated_cells += length

    def get_stats(self) -> Stats:
        return self.stats.get_stats()


class SRAM(Memory):
    """SRAM registers file component for the Core"""

    def __init__(self, core_config: CoreConfig = None):
        self.core_config = core_config or CoreConfig()
        size = self.core_config.dataMem_size
        super().__init__(size)

        # Initialize stats
        self.stats.config = self.core_config
        self.stats.memory_type = "SRAM"


class DRAM(Memory):
    """DRAM array component for the Tile"""

    def __init__(self, tile_config: TileConfig = None):
        self.tile_config = tile_config or TileConfig()
        size = self.tile_config.edram_size
        super().__init__(size)

        # Initialize stats
        self.stats.config = self.tile_config
        self.stats.memory_type = "DRAM"
