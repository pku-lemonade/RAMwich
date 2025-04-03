import logging
from typing import List, Dict, Any
from .tile import Tile
from pydantic import BaseModel, Field

class NodeStats(BaseModel):
    operations: int = Field(default=0, description="Total number of operations")
    load_operations: int = Field(default=0, description="Number of load operations")
    set_operations: int = Field(default=0, description="Number of set operations")
    alu_operations: int = Field(default=0, description="Number of ALU operations")
    mvm_operations: int = Field(default=0, description="Number of MVM operations")
    total_execution_time: float = Field(default=0, description="Total execution time")

    def get_stats(self, node_id: int, include_components: bool = True, tiles=None) -> Dict[str, Any]:
        """Get statistics for this Node and optionally its components"""
        result = {
            'node_id': node_id,
            'stats': self.dict()
        }

        if include_components and tiles:
            result['tiles'] = [
                tile.get_stats(include_components)
                for tile in tiles
            ]

        return result

class Node:
    """
    Node in the RAMwich architecture, containing multiple tiles.
    """
    def __init__(self, id: int, tiles: List[Tile]):
        self.id = id
        self.tiles = tiles
        self.stats = NodeStats()

    def __repr__(self):
        return f"Node({self.id}, tiles={len(self.tiles)})"

    def get_tile(self, tile_id):
        """Get a specific tile by ID"""
        raise self.tiles

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
        """Get statistics for this Node and optionally its components"""
        return self.stats.get_stats(self.id, include_components, self.tiles)

    def run(self, simulator, env):
        """Execute operations for all tiles in this node"""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Starting operations for node {self.id}")

        # Start all tiles in parallel
        processes = []
        for tile in self.tiles:
            processes.append(env.process(tile.run(simulator, env)))

        # Wait for all tiles to complete
        yield env.all_of(processes)

        logger.info(f"Completed all operations for node {self.id}")
