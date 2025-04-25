from typing import Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import CoreConfig, TileConfig
from ..stats import Stats


class MemoryStats(BaseModel):
    """Statistics for Memory operations"""

    # Universal metrics
    unit_energy_consumption_read: float = Field(default=0.0, description="Energy consumption for each convertion in pJ")
    unit_energy_consumption_write: float = Field(
        default=0.0, description="Energy consumption for each convertion in pJ"
    )
    leakage_energy_per_cycle: float = Field(default=0.0, description="Leakage energy consumption for 1 cycle in pJ")
    area = float = Field(default=0.0, description="Area in mm^2")

    # Memory specific metrics
    memory_type: str = Field(default="", description="Type of memory (SRAM/DRAM)")
    read_operations: int = Field(default=0, description="Number of read operations")
    read_cells: int = Field(default=0, description="Number of read cells")
    write_operations: int = Field(default=0, description="Number of write operations")
    write_cells: int = Field(default=0, description="Number of written cells")
    total_operations: int = Field(default=0, description="Total number of operations")
    total_operated_cells: int = Field(default=0, description="Total number of operated cells")

    def get_stats(self) -> Stats:
        """Convert MemoryStats to general Stats object"""
        stats = Stats()

        # Map ADC metrics to Stat object
        if self.memory_type == "SRAM":
            stats.dynamic_energy = (
                self.unit_energy_consumption_read * self.read_cells
                + self.unit_energy_consumption_write * self.write_cells
            )
        elif self.memory_type in ["DRAM", "Input Register Array", "Output Register Array"]:
            stats.dynamic_energy = (
                self.unit_energy_consumption_read * self.read_operations
                + self.unit_energy_consumption_write * self.write_operations
            )
        stats.leakage_energy = self.leakage_energy_per_cycle
        stats.area = self.area

        stats.increment_component_count(self.memory_type, self.conversions)
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
        self.stats.read_operations += 1
        self.stats.read_cells += length
        self.stats.total_operations += 1
        self.stats.total_operated_cells += length

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
        self.stats.write_operations += 1
        self.stats.write_cells += length
        self.stats.total_operations += 1
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
        self.stats.Memory_type = "SRAM"
        self.stats.unit_energy_consumption_read = self.core_config.dataMem_pow_dyn
        self.stats.unit_energy_consumption_write = self.core_config.dataMem_pow_dyn
        self.stats.leakage_energy_per_cycle = self.core_config.dataMem_pow_leak
        self.stats.area = self.core_config.dataMem_area


class DRAM(Memory):
    """DRAM array component for the Tile"""

    def __init__(self, tile_config: TileConfig = None):
        self.tile_config = tile_config or TileConfig()
        size = self.tile_config.edram_size
        super().__init__(size)

        # Initialize stats
        self.stats.Memory_type = "DRAM"
        self.stats.unit_energy_consumption_read = self.tile_config.edram_pow_dyn
        self.stats.unit_energy_consumption_write = self.tile_config.edram_pow_dyn
        self.stats.leakage_energy_per_cycle = self.tile_config.edram_pow_leak
        self.stats.area = self.tile_config.edram_area
