from typing import List
from tile import Tile

class Node:
    """
    Node in the RAMwich architecture, containing multiple tiles.
    """
    def __init__(self, id: int, tiles: List[Tile]):
        self.id = id
        self.tiles = tiles
        self.stats = {
            'operations': 0,
            'load_operations': 0,
            'set_operations': 0,
            'alu_operations': 0,
            'mvm_operations': 0,
            'total_execution_time': 0
        }

    def __repr__(self):
        return f"Node({self.id}, tiles={len(self.tiles)})"

    def get_tile(self, tile_id):
        """Get a specific tile by ID"""
        for tile in self.tiles:
            if tile.id == tile_id:
                return tile
        raise ValueError(f"Tile with ID {tile_id} not found")

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
        """Get statistics for this Node and optionally its components"""
        result = {
            'node_id': self.id,
            'stats': self.stats.copy()
        }

        if include_components:
            result['tiles'] = [
                tile.get_stats(include_components)
                for tile in self.tiles
            ]

        return result

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
