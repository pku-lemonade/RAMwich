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
        raise self.tiles[tile_id]

    def get_stats(self) -> Stat:
        """Get statistics for this Node and optionally its components"""
        return self.stats.get_stats(components=self.tiles)

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
