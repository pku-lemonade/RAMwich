import logging
from typing import List

import numpy as np
import simpy

from .blocks.dram_controller import DRAMController
from .blocks.memory import DRAM
from .blocks.router import Network, Router
from .config import Config
from .core import Core
from .ops import Halt, Recv, Send, TileOpType
from .stats import Stats

logger = logging.getLogger(__name__)


class Tile:
    """
    Tile in the RAMwich architecture, containing multiple cores.
    """

    def __init__(self, id: int, parent, config: Config = None):
        self.id = id
        self.parent = parent
        self.config = config or Config()
        self.tile_config = self.config.tile_config
        self.operations: List[TileOpType] = []

        # Initialize components
        self.edram = DRAM(self.tile_config)
        self.dram_controller = DRAMController(dram=self.edram, tile_config=self.tile_config)
        self.router = Router(network=parent.network, id=self.id, config=self.config)

        # Initialize cores
        self.cores = [Core(id=i, parent=self, config=self.config) for i in range(self.config.num_cores_per_tile)]

        # Initialize stats
        self.stats = Stats()

        # Initialize simulation timing attributes
        self.start_time = 0
        self.active_cycles = 0

    def __repr__(self):
        return f"Tile({self.id}, cores={len(self.cores)})"

    def get_core(self, core_id):
        """Get a specific core by ID"""
        return self.cores[core_id]

    def execute_send(self, op: Send):
        """Execute a Send operation"""

        # First read the data from the source
        read_event = self.dram_controller.submit_read_request(
            core_id=-1, start=op.mem_addr, batch_size=op.width, num_batches=op.vec
        )
        yield read_event
        data = read_event.value

        # Reshape the data to match the send operation
        data = data.reshape((op.vec, op.width))

        for i in range(op.vec):
            # Add the package to router's send queue
            yield self.env.process(self.router.add_send_packet(target=op.target_tile, data=data[i]))

        return True

    def execute_receive(self, op: Recv):
        """Execute a Receive operation"""

        # creata a data buffer to store the received data
        received_data = np.zeros((op.vec, op.width), dtype=np.int32)
        for i in range(op.vec):
            # Wait for the data to be available in the router's receive buffer
            data = yield self.env.process(self.router.read_packet(source=op.source_tile))

            # Validate the data size matches the expected size
            if len(data) != op.width:
                logger.error(f"Tile {self.id}: Data size mismatch for receive operation at time {self.env.now}")
                return False

            # Copy the data to the received_data buffer
            np.copyto(received_data[i], data)

        # Write the received data to the DRAM
        yield self.dram_controller.submit_write_request(core_id=-1, start=op.mem_addr, data=received_data)

        return True

    def execute_halt(self, op: Halt):
        """Execute a Halt operation"""

        # wait for all cores, router and dram controller to finish
        if self.core_processes:
            yield self.env.all_of(self.core_processes)
        # wait for router to stop
        yield self.env.process(self.router.stop_after_all_packets_sent())

        # DRAM controller will be safe to stop since all cores are halted
        # Also no more send and receive operations will be submitted
        self.dram_controller.stop()

        return True

    def run(self, env):
        """Execute operations for this tile and its cores"""
        logger.info(f"Tile {self.id} starting execution at time {env.now}")

        self.start_time = env.now

        self.env = env

        self.core_processes = [env.process(core.run(env)) for core in self.cores]
        self.dram_controller.run(env)
        self.router.run(env)

        for op in self.operations:
            success = yield env.process(op.accept(self))
            if not success:
                logger.warning(f"Tile {self.id}: Operation {op} failed at time {env.now}")
            else:
                logger.debug(f"Tile {self.id}: Operation {op} completed at time {env.now}")

        self.active_cycles = env.now - self.start_time

        logger.info(f"Tile {self.id} finished execution at time {env.now}")

    def get_stats(self) -> Stats:
        """Get statistics for this Tile and its components"""
        # For I/O tiles, we only need to return the stats of the router
        if self.id in [0, 1]:
            return self.stats.get_stats([self.router])

        # first add pseudo compoents stats
        # Tile Control Unit
        self.stats.increment_component_activation("Tile Control Unit", self.active_cycles)
        self.stats.increment_component_dynamic_energy(
            "Tile Control Unit", self.active_cycles * self.tile_config.tcu_pow
        )
        # self.stats.increment_component_leakage_energy("Tile Control Unit", self.tile_config.tcu_pow_leak)
        self.stats.increment_component_area("Tile Control Unit", self.tile_config.tcu_area)

        # Tile instruction memory
        self.stats.increment_component_activation("Tile instruction memory", len(self.operations))
        self.stats.increment_component_dynamic_energy(
            "Tile instruction memory", len(self.operations) * self.tile_config.instrnMem_pow_dyn
        )
        self.stats.increment_component_leakage_energy("Tile instruction memory", self.tile_config.instrnMem_pow_leak)
        self.stats.increment_component_area("Tile instruction memory", self.tile_config.instrnMem_area)

        # then add stats from all components except the cores
        self.stats.get_stats([self.router, self.dram_controller, self.edram])

        # Calculate the leakage energy all components except the cores based on active cycles
        self.stats.calculate_leakage_energy(self.active_cycles)

        # Add the stats of all cores
        return self.stats.get_stats(self.cores)
