import logging
from typing import List, Dict, Any
from .tile import Tile
from .stats import Stat

logger = logging.getLogger(__name__)

class Node:
    """
    Node in the RAMwich architecture, containing multiple tiles.
    """
    def __init__(self, id: int, tiles: List[Tile]):
        self.id = id
        self.tiles = tiles
        self.stats = Stat()

    def __repr__(self):
        return f"Node({self.id}, tiles={len(self.tiles)})"

    def get_tile(self, tile_id):
        """Get a specific tile by ID"""
        raise self.tiles

    def update_stats(self, op_type):
        """Update operation count statistics"""
        self.stats.operations += 1

    def update_execution_time(self, execution_time):
        """Update the execution time statistics"""
        self.stats.latency += execution_time

    def get_stats(self, include_components=True):
        """Get statistics for this Node and optionally its components"""
        return self.stats.get_stats(
            component_id=self.id,
            component_type="node",
            components=self.tiles if include_components else None
        )

    def run(self, simulator, env):
        """Execute operations for all tiles in this node"""
        logger.info(f"Starting operations for node {self.id}")

        # Start all tiles in parallel
        processes = []
        for tile in self.tiles:
            processes.append(env.process(tile.run(simulator, env)))

        # Wait for all tiles to complete
        yield env.all_of(processes)

        logger.info(f"Completed all operations for node {self.id}")
