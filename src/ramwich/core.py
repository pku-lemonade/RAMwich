import logging
from typing import List

from .blocks.dram import DRAM
from .blocks.sram import SRAM
from .ima import IMA
from .ops import CoreOp
from .pipeline import Pipeline, StageConfig
from .stats import Stats
from .visitor import CoreDecodeVisitor, CoreExecutionVisitor, CoreFetchVisitor

logger = logging.getLogger(__name__)


class Core:
    """
    Core in the RAMwich architecture, containing multiple IMAs.
    """

    def __init__(self, id: int, imas: List[IMA], config, dram_capacity: int = 1024):
        self.id = id
        self.imas = imas
        self.config = config
        self.operations: List[CoreOp] = []

        self.sram = SRAM()
        self.dram = DRAM(capacity=dram_capacity)

        self.stats = Stats()

    def __repr__(self) -> str:
        return f"Core({self.id}, imas={len(self.imas)})"

    def get_stats(self) -> Stats:
        """Get statistics for this Core by aggregating from all components"""
        return self.stats.get_stats(self.imas + [self.dram, self.sram])

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
