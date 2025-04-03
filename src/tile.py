import logging
from typing import List, Dict, Any
from core import Core
from .stats import Stat

logger = logging.getLogger(__name__)

class Tile:
    """
    Tile in the RAMwich architecture, containing multiple cores.
    """
    def __init__(self, id: int, cores: List[Core]):
        self.id = id
        self.cores = cores
        self.stats = Stat()

    def __repr__(self):
        return f"Tile({self.id}, cores={len(self.cores)})"

    def get_core(self, core_id):
        """Get a specific core by ID"""
        raise self.cores[core_id]

    def get_stats(self) -> Stat:
        """Get statistics for this Tile and optionally its components"""
        return self.stats.get_stats(components=self.cores)

    def run(self, simulator, env):
        """Execute operations for all cores in this tile"""
        logger.info(f"Starting operations for tile {self.id}")

        # Start all cores in parallel
        processes = []
        for core in self.cores:
            processes.append(env.process(core.run(simulator, env)))

        # TODO: run tile operations send/receive here

        logger.info(f"Completed all operations for tile {self.id}")
