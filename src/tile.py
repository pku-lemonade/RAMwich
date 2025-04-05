import logging
from typing import List, Dict, Any, TYPE_CHECKING
from .core import Core # Keep Core import for type hint in __init__
from .stats import Stats # Use Stats consistently
from .ops import Send, Recv, TileOpType # Import TileOp types

logger = logging.getLogger(__name__)

class Tile:
    """
    Tile in the RAMwich architecture, containing multiple cores.
    """
    def __init__(self, id: int, cores: List[Core], config):
        self.id = id
        self.cores = cores
        self.config = config
        self.operations: List[TileOpType] = []
        self.stats = Stats()

    def __repr__(self):
        return f"Tile({self.id}, cores={len(self.cores)})"

    def get_core(self, core_id):
        """Get a specific core by ID"""
        raise self.cores[core_id]

    def get_stats(self) -> Stats:
        """Get statistics for this Tile and optionally its components"""
        # Start with the tile's own stats (op_counts)
        aggregated_stats = self.stats.copy(deep=True)

        # Aggregate stats from cores
        for core in self.cores:
            component_stats = core.get_stats()
            aggregated_stats.latency += component_stats.latency
            aggregated_stats.energy += component_stats.energy
            aggregated_stats.area += component_stats.area
            # Aggregate op_counts carefully
            for op_type, count in component_stats.op_counts.items():
                aggregated_stats.increment_op_count(op_type, count)

        return aggregated_stats

    def execute_send(self, op: Send) -> bool:
        """Execute a Send operation (placeholder)"""
        logger.debug(f"Tile {self.id} executing Send: {op}")
        # Placeholder: Simulate sending data via NoC
        # Actual implementation would interact with NoC component
        # Need to determine execution time based on config/NoC simulation
        self.stats.increment_op_count('send')
        # self.stats.latency += send_latency # Update latency when known
        return True # Assume success for now

    def execute_receive(self, op: Recv) -> bool:
        """Execute a Receive operation (placeholder)"""
        logger.debug(f"Tile {self.id} executing Receive: {op}")
        # Placeholder: Simulate receiving data via NoC
        # Actual implementation would interact with NoC component
        # Need to determine execution time based on config/NoC simulation
        self.stats.increment_op_count('receive')
        # self.stats.latency += recv_latency # Update latency when known
        return True # Assume success for now

    def run(self, env):
        """Execute operations for this tile and its cores"""
        logger.info(f"Tile {self.id} starting execution at time {env.now}")

        # Start core processes
        core_processes = [env.process(core.run(env)) for core in self.cores]

        # Process tile-level operations
        for op in self.operations:
            # Get execution time from config
            exec_time = 1 # Default
            if op.type == 'send':
                exec_time = self.config.noc_config.noc_intra_lat
            elif op.type == 'receive':
                exec_time = self.config.noc_config.noc_intra_lat

            yield env.timeout(exec_time)
            success = op.accept(self)
            if not success:
                logger.warning(f"Tile {self.id}: Operation {op} failed at time {env.now}")
            else:
                logger.debug(f"Tile {self.id}: Operation {op} completed at time {env.now}")
                self.stats.latency += exec_time

        # Wait for all core processes to complete (if any started)
        if core_processes:
            yield env.all_of(core_processes)

        logger.info(f"Tile {self.id} finished execution at time {env.now}")
