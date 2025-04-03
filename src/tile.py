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

    def update_stats(self, op_type):
        """Update operation count statistics"""
        self.stats.operations += 1

    def update_execution_time(self, execution_time):
        """Update the execution time statistics"""
        self.stats.latency += execution_time

    def get_stats(self, include_components=True):
        """Get statistics for this Tile and optionally its components"""
        return self.stats.get_stats(
            component_id=self.id,
            component_type="tile",
            components=self.cores if include_components else None
        )

    def run(self, simulator, env):
        """Execute operations for all cores in this tile"""
        logger.info(f"Starting operations for tile {self.id}")

        # Start all cores in parallel
        processes = []
        for core in self.cores:
            processes.append(env.process(core.run(simulator, env)))

        # Wait for all cores to complete
        yield env.all_of(processes)

        logger.info(f"Completed all operations for tile {self.id}")
