import logging
from typing import List

from .blocks.router import Network
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
        self.network = Network()
        self.tiles = [Tile(id=i, parent=self, config=config) for i in range(config.num_tiles_per_node)]

        # Initialize stats
        self.stats = Stats()

    def __repr__(self):
        return f"Node({self.id}, tiles={len(self.tiles)})"

    def get_tile(self, tile_id):
        """Get a specific tile by ID"""
        return self.tiles[tile_id]

    def run(self, env):
        """Execute operations for all tiles in this node"""
        logger.info(f"Starting operations for node {self.id}")

        self.env = env

        processes = []
        for tile in self.tiles:
            processes.append(env.process(tile.run(env)))

        yield env.all_of(processes)

        logger.info(f"Completed all operations for node {self.id}")

    def get_stats(self) -> Stats:
        """Get statistics for this Node and its components"""
        # first add pseudo stats
        self.stats.increment_component_area("NOC inter", self.config.noc_config.noc_inter_area)
        self.stats.increment_component_area(
            "NOC intra",
            self.config.noc_config.noc_intra_area * self.config.num_tiles_per_node / self.config.noc_config.num_port,
        )
        self.stats.increment_component_leakage_energy(
            "NOC",
            self.config.noc_config.noc_intra_pow_leak
            * self.config.num_tiles_per_node
            / self.config.noc_config.num_port,
        )
        return self.stats.get_stats(components=self.tiles)
