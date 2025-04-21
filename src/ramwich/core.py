import logging
from typing import List

import numpy as np
from numpy.typing import NDArray

from .blocks.dram_controller import DRAMController
from .blocks.memory import SRAM
from .blocks.vfu import VFU
from .config import Config
from .mvmu import MVMU
from .ops import CoreOp
from .pipeline import Pipeline, StageConfig
from .stats import Stats
from .visitor import CoreDecodeVisitor, CoreExecutionVisitor, CoreFetchVisitor

logger = logging.getLogger(__name__)


class Core:
    """
    Core in the RAMwich architecture, containing multiple MVMUs.
    """

    def __init__(self, id: int, dram_controller: DRAMController, config: Config = None):
        self.id = id
        self.config = config or Config()
        self.core_config = self.config.core_config
        self.operations: List[CoreOp] = []

        # Initialize some useful parameters
        # Address 0 to num_mvmus_per_core * xbar_size - 1 are the MVMU input registers
        # Address num_mvmus_per_core * xbar_size to num_mvmus_per_core * xbar_size * 2 - 1 are the MVMU output registers
        # From num_mvmus_per_core * xbar_size * 2 and above are core cache
        self.mvmu_outreg_start = self.config.mvmu_config.xbar_config.xbar_size * self.config.num_mvmus_per_core
        self.cache_start = self.mvmu_outreg_start * 2
        self.total_registers = self.cache_start + self.config.core_config.dataMem_size

        # Initialize components
        self.cache = SRAM(self.core_config)
        self.vfu = VFU(self.config)
        self.dram_controller = dram_controller

        # Initialize MVMUs
        self.mvmus = [MVMU(id=i, config=self.config) for i in range(self.config.num_mvmus_per_core)]

        # Initialize stats
        self.stats = Stats()

    def __repr__(self) -> str:
        return f"Core({self.id}, mvmus={len(self.mvmus)})"

    def get_mvmu(self, mvmu_id: int):
        """Get the MVMU instance by ID"""

        # Validate MVMU ID
        if mvmu_id < 0 or mvmu_id >= len(self.mvmus):
            raise IndexError(f"MVMU ID {mvmu_id} out of range")

        return self.mvmus[mvmu_id]

    def get_stats(self) -> Stats:
        """Get statistics for this Core by aggregating from all components"""
        return self.stats.get_stats(self.mvmus + [self.dram, self.sram])

    def _overlaps_output_registers(self, start: int, end: int) -> bool:
        """Check if address range overlaps with MVMU output registers"""
        # Check if the range [start, end) overlaps with output register range
        return (
            (start >= self.mvmu_outreg_start and start < self.cache_start)
            or (end > self.mvmu_outreg_start and end <= self.cache_start)
            or (start <= self.mvmu_outreg_start and end >= self.cache_start)
        )

    def _get_mvmu_id_from_address(self, start: int, end: int) -> int:
        """Get the MVMU ID for a given address range"""
        # Validate input
        mvmu_id = start // self.config.mvmu_config.xbar_config.xbar_size
        mvmu_id_end = (end - 1) // self.config.mvmu_config.xbar_config.xbar_size
        if mvmu_id != mvmu_id_end:
            raise IndexError(f"Address spans multiple MVMUs ({mvmu_id}, {mvmu_id_end})")

        return mvmu_id % self.config.num_mvmus_per_core

    def write_to_register(self, start: int, data: NDArray[np.int32]):
        """Write data to the register file of the MVMU."""

        length = len(data)
        end = start + length

        # Validate input
        if start < 0 or end > self.total_registers:
            raise IndexError(f"Write operation out of range ({start}, {length})")

        if self._overlaps_output_registers(start, end):
            raise IndexError(f"Write operation to MVMU output register ({start}, {length}) is not allowed")

        # Depending on address, write to the appropriate register
        if start >= self.cache_start:
            # Write to cache
            internal_start = start - self.cache_start
            self.cache.write(internal_start, data)
        else:
            # Write to MVMU input registers
            mvmu_id = self._get_mvmu_id_from_address(start, end)
            internal_start = start % self.config.mvmu_config.xbar_config.xbar_size

            # Write to the input register of the MVMU
            self.mvmus[mvmu_id].write_to_inreg(internal_start, data)

    def read_from_register(self, start: int, length: int) -> NDArray[np.int32]:
        """Read data from the register file of the MVMU."""
        end = start + length

        # Validate input
        if start < 0 or end > self.total_registers:
            raise IndexError(f"Read operation out of range ({start}, {length})")

        if start < self.mvmu_outreg_start:
            raise IndexError(f"Read operation from MVMU input register ({start}, {length}) is not allowed")

        # Depending on address, read from the appropriate register
        if start >= self.cache_start:
            # Read from cache
            internal_start = start - self.cache_start
            return self.cache.read(internal_start, length)
        else:
            # Read from MVMU output registers
            mvmu_id = self._get_mvmu_id_from_address(start, end)
            internal_start = start % self.config.mvmu_config.xbar_config.xbar_size

            # Read from the input register of the MVMU
            return self.mvmus[mvmu_id].read_from_outreg(internal_start, length)

    def run(self, env):
        """
        Execute all operations assigned to this core using a pipeline.
        This method should be called as a SimPy process.
        """
        logger.info(f"Core {self.id} starting execution at time {env.now}")

        # Create pipeline stages
        pipeline_config = [
            StageConfig("fetch", CoreFetchVisitor(self.config)),
            StageConfig("decode", CoreDecodeVisitor(self.config)),
            StageConfig("execute", CoreExecutionVisitor(self)),
        ]

        pipeline = Pipeline(env, pipeline_config)
        pipeline.run()

        # Feed instructions into pipeline
        for op in self.operations:
            yield pipeline.put(op)

        logger.info(f"Core {self.id} finished execution at time {env.now}")
