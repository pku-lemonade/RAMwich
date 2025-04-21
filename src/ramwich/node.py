import logging
from typing import List

from .config import Config
from .stats import Stats
from .tile import Tile

logger = logging.getLogger(__name__)


class Node:
    """
    Node in the RAMwich architecture, containing multiple tiles.
    """

    def __init__(self, id: int, config: Config = None):
        self.id = id
        self.config = config or Config()
        self.tiles = [Tile(id=i, config=config) for i in range(config.num_tiles_per_node)]

        # Initialize stats
        self.stats = Stats()

    def __repr__(self):
        return f"Node({self.id}, tiles={len(self.tiles)})"

    def get_tile(self, tile_id):
        """Get a specific tile by ID"""
        return self.tiles[tile_id]

    def get_stats(self) -> Stats:
        """Get statistics for this Node and optionally its components"""
        return self.stats.get_stats(components=self.tiles)

    def run(self, env):
        """Execute operations for all tiles in this node"""
        logger.info(f"Starting operations for node {self.id}")

        processes = []
        for tile in self.tiles:
            processes.append(env.process(tile.run(env)))

        yield env.all_of(processes)

        logger.info(f"Completed all operations for node {self.id}")
