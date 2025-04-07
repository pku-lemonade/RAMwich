import logging
from typing import List

from .core import Core
from .ops import Recv, Send, TileOpType
from .stats import Stats

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
        aggregated_stats = self.stats.copy(deep=True)

        for core in self.cores:
            component_stats = core.get_stats()
            aggregated_stats.latency += component_stats.latency
            aggregated_stats.energy += component_stats.energy
            aggregated_stats.area += component_stats.area
            for op_type, count in component_stats.op_counts.items():
                aggregated_stats.increment_op_count(op_type, count)

        return aggregated_stats

    def execute_send(self, op: Send) -> bool:
        """Execute a Send operation (placeholder)"""
        logger.debug(f"Tile {self.id} executing Send: {op}")
        self.stats.increment_op_count("send")
        return True

    def execute_receive(self, op: Recv) -> bool:
        """Execute a Receive operation (placeholder)"""
        logger.debug(f"Tile {self.id} executing Receive: {op}")
        self.stats.increment_op_count("receive")
        return True

    def run(self, env):
        """Execute operations for this tile and its cores"""
        logger.info(f"Tile {self.id} starting execution at time {env.now}")

        core_processes = [env.process(core.run(env)) for core in self.cores]

        for op in self.operations:
            exec_time = 1
            if op.type == "send":
                exec_time = self.config.noc_config.noc_intra_lat
            elif op.type == "receive":
                exec_time = self.config.noc_config.noc_intra_lat

            yield env.timeout(exec_time)
            success = op.accept(self)
            if not success:
                logger.warning(f"Tile {self.id}: Operation {op} failed at time {env.now}")
            else:
                logger.debug(f"Tile {self.id}: Operation {op} completed at time {env.now}")
                self.stats.latency += exec_time

        if core_processes:
            yield env.all_of(core_processes)

        logger.info(f"Tile {self.id} finished execution at time {env.now}")
