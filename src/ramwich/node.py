import logging

from .blocks.router import Network
from .config import Config
from .stats import StatsDict
from .tile import Tile

logger = logging.getLogger(__name__)


class Node:
    """
    Node in the RAMwich architecture, containing multiple tiles.
    """

    def __init__(self, id: int, config: Config):
        self.id = id
        self.config = config
        self.network = Network(self.config)
        self.tiles = [Tile(id=i, parent=self, config=config) for i in range(config.num_tiles_per_node)]

        self.network_busy_cycles = 0

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

    def reset(self):
        """Reset the node and its components"""
        for tile in self.tiles:
            tile.reset()

    def get_stats(self) -> StatsDict:
        """Get statistics for this Node and its components"""
        stats_dict = StatsDict()

        # first add NOC stats
        stats_dict.merge(self.network.get_stats())

        # Calculate the leakage energy for the NOC
        stats_dict.update_leakage_energy(self.network.get_queue_busy_cycles())

        # Add the stats from each tile
        for tile in self.tiles:
            stats_dict.merge(tile.get_stats())

        # Calculate the dynamic energy for Router intranode send since it is base on the busy cycles
        intra_node_dynamic_energy = self.config.noc_config.noc_intra_pow_dyn * self.network.get_queue_busy_cycles()
        stats_dict["Router"].dynamic_energy += intra_node_dynamic_energy

        return stats_dict
