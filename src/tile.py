from typing import List
from core import Core

class Tile:
    """
    Tile in the RAMwich architecture, containing multiple cores.
    """
    def __init__(self, id: int, cores: List[Core]):
        self.id = id
        self.cores = cores
        self.stats = {
            'operations': 0,
            'load_operations': 0,
            'set_operations': 0,
            'alu_operations': 0,
            'mvm_operations': 0,
            'total_execution_time': 0
        }

    def __repr__(self):
        return f"Tile({self.id}, cores={len(self.cores)})"

    def get_core(self, core_id):
        """Get a specific core by ID"""
        for core in self.cores:
            if core.id == core_id:
                return core
        raise ValueError(f"Core with ID {core_id} not found")

    def update_stats(self, op_type):
        """Update operation count statistics"""
        self.stats['operations'] += 1
        op_key = f"{op_type}_operations"
        if op_key in self.stats:
            self.stats[op_key] += 1

    def update_execution_time(self, execution_time):
        """Update the execution time statistics"""
        self.stats['total_execution_time'] += execution_time

    def get_stats(self, include_components=True):
        """Get statistics for this Tile and optionally its components"""
        result = {
            'tile_id': self.id,
            'stats': self.stats.copy()
        }

        if include_components:
            result['cores'] = [
                core.get_stats(include_components)
                for core in self.cores
            ]

        return result

    def run(self, simulator, env):
        """Execute operations for all cores in this tile"""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Starting operations for tile {self.id}")

        # Start all cores in parallel
        processes = []
        for core in self.cores:
            processes.append(env.process(core.run(simulator, env)))

        # Wait for all cores to complete
        yield env.all_of(processes)

        logger.info(f"Completed all operations for tile {self.id}")
