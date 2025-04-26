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
        self.network = Network(self.config)
        self.tiles = [Tile(id=i, parent=self, config=config) for i in range(config.num_tiles_per_node)]

        self.network_busy_cycles = 0

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
        self.network.start_tracking(env)

        processes = []
        for tile in self.tiles:
            processes.append(env.process(tile.run(env)))

        yield env.all_of(processes)
        self.network.stop_tracking()

        logger.info(f"Completed all operations for node {self.id}")

    def get_stats(self) -> Stats:
        """Get statistics for this Node and its components"""
        # first add NOC stats
        self.stats.get_stats(components=[self.network])

        # Calculate the leakage energy for the NOC
        self.stats.calculate_leakage_energy(self.network.get_queue_busy_cycles())

        # Add the stats from each tile
        self.stats.get_stats(components=self.tiles)

        # Calculate the dynamic energy for the NOC
        internode_packets = self.stats.components_activation_count["Router send internode"]
        self.stats.increment_component_dynamic_energy(
            "Router send internode",
            self.config.noc_config.noc_inter_pow_dyn * internode_packets / 12,  # Align with PUMA
        )
        self.stats.increment_component_dynamic_energy(
            "Router send intranode", self.config.noc_config.noc_intra_pow_dyn * self.network.get_queue_busy_cycles()
        )

        return self.stats
