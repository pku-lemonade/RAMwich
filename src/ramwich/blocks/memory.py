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

    def reset(self):
        """Reset all statistics to zero"""
        self.read_operations = 0
        self.read_cells = 0
        self.write_operations = 0
        self.write_cells = 0
        self.total_operations = 0
        self.total_operated_cells = 0

    def get_stats(self) -> StatsDict:
        """Convert MemoryStats to general Stats object"""

        # Map Memory metrics to StatsDict object
        if self.memory_type == "SRAM cache":
            stats = Stats(
                activation_count=self.total_operated_cells,
                dynamic_energy=self.config.dataMem_pow_dyn * self.total_operated_cells,
                leakage_energy=self.config.dataMem_pow_leak,
                area=self.config.dataMem_area,
            )
            return StatsDict({"SRAM Cache": stats})

        elif self.memory_type == "SRAM storage":
            stats = Stats(
                activation_count=self.total_operated_cells,
                dynamic_energy=self.config.storage_pow_dyn * self.total_operated_cells,
                leakage_energy=self.config.storage_pow_leak,
                area=self.config.storage_area,
            )
            return StatsDict({"SRAM storage": stats})

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

        # Initialize memory cells as raw 32-bit words; interpretation handled on read/write
        self.cells = np.zeros(self.size, dtype=np.uint32)
        # Track whether each cell currently stores a float (True) or int (False)
        self.type_bits = np.zeros(self.size, dtype=np.bool_)

        # Initialize stats
        self.stats = MemoryStats()

    def read(self, start: int, length: int, batch: int = 1):
        """Read a block of registers from SRAM.

        The values are automatically converted back to their stored precision
        (float32 or int32) using the type metadata captured during writes. If a
        mixture of types exists inside the requested span, the result is
        promoted to float32 to avoid information loss.
        """

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

        raw = self.cells[start:end].copy()
        type_slice = self.type_bits[start:end]

        if np.all(type_slice):
            return raw.view(np.float32)
        if not np.any(type_slice):
            return raw.view(np.int32)

        # Mixed types: promote to float32 for safe interpretation
        promoted = raw.view(np.float32)
        return promoted

    def reset(self):
        """Reset the memory cells and statistics"""
        self.cells.fill(0)
        self.type_bits.fill(False)
        self.stats.reset()

    def write(
        self,
        start: int,
        values: Union[NDArray[np.int32], NDArray[np.float32], int, float],
        batch: int = 1,
    ):
        """Write values to a block of registers in SRAM.

        Accepts either 32-bit integer or 32-bit floating-point inputs. Values are stored as
        raw 32-bit words, preserving exact bit patterns for both representations.
        """

        # Normalize input to numpy array
        if isinstance(values, (int, float)):
            values = np.array([values], dtype=np.float32 if isinstance(values, float) else np.int32)
        else:
            values = np.asarray(values)

        if np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float32, copy=False)
            stored = values.view(np.uint32)
            is_float = True
        elif np.issubdtype(values.dtype, np.integer):
            values = values.astype(np.int32, copy=False)
            stored = values.view(np.uint32)
            is_float = False
        else:
            raise TypeError("Memory write supports only 32-bit integer or float data")

        length = len(values)
        end = start + length

        # Validate input
        if start < 0 or end > self.size:
            raise IndexError(f"Write operation out of range ({start}, {length})")

        # Write values
        self.cells[start:end] = stored
        self.type_bits[start:end] = is_float

        # Update stats
        self.stats.write_operations += batch
        self.stats.write_cells += length
        self.stats.total_operations += batch
        self.stats.total_operated_cells += length

    def get_stats(self) -> StatsDict:
        return self.stats.get_stats()


class SRAM(Memory):
    """SRAM registers file component for the Core"""

    def __init__(self, core_config: CoreConfig, sram_type: str = "cache"):
        self.core_config = core_config
        if sram_type == "cache":
            size = self.core_config.dataMem_size
        elif sram_type == "storage":
            size = self.core_config.storage_size
        else:
            raise ValueError(f"Unknown SRAM type: {sram_type}")

        super().__init__(size)

        # Initialize stats
        self.stats.config = self.core_config
        self.stats.memory_type = "SRAM " + sram_type


class DRAM(Memory):
    """DRAM array component for the Tile"""

    def __init__(self, tile_config: TileConfig):
        self.tile_config = tile_config
        size = self.tile_config.edram_size
        super().__init__(size)

        # Initialize stats
        self.stats.config = self.tile_config
        self.stats.memory_type = "DRAM"
