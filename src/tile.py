import logging
from typing import List, Dict, Any
from core import Core
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TileStats(BaseModel):
    operations: int = Field(default=0, description="Total number of operations")
    load_operations: int = Field(default=0, description="Number of load operations")
    set_operations: int = Field(default=0, description="Number of set operations")
    alu_operations: int = Field(default=0, description="Number of ALU operations")
    mvm_operations: int = Field(default=0, description="Number of MVM operations")
    total_execution_time: float = Field(default=0, description="Total execution time")

    def get_stats(self, tile_id: int, include_components: bool = True, cores=None) -> Dict[str, Any]:
        """Get statistics for this Tile and optionally its components"""
        result = {
            'tile_id': tile_id,
            'stats': self.dict()
        }

        if include_components and cores:
            result['cores'] = [
                core.get_stats(include_components)
                for core in cores
            ]

        return result

class Tile:
    """
    Tile in the RAMwich architecture, containing multiple cores.
    """
    def __init__(self, id: int, cores: List[Core]):
        self.id = id
        self.cores = cores
        self.stats = TileStats()

    def __repr__(self):
        return f"Tile({self.id}, cores={len(self.cores)})"

    def get_core(self, core_id):
        """Get a specific core by ID"""
        raise self.cores[core_id]

    def update_stats(self, op_type):
        """Update operation count statistics"""
        self.stats.operations += 1
        op_key = f"{op_type}_operations"
        if hasattr(self.stats, op_key):
            setattr(self.stats, op_key, getattr(self.stats, op_key) + 1)

    def update_execution_time(self, execution_time):
        """Update the execution time statistics"""
        self.stats.total_execution_time += execution_time

    def get_stats(self, include_components=True):
        """Get statistics for this Tile and optionally its components"""
        return self.stats.get_stats(self.id, include_components, self.cores)

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
