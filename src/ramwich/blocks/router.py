import numpy as np
import simpy
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import Config
from ..stats import Stats


class RouterStats(BaseModel):
    """Statistics for DRAM operations"""

    packets_created: int = Field(default=0, description="Number of packets created")
    packets_sent: int = Field(default=0, description="Number of packets sent")
    packets_received: int = Field(default=0, description="Number of packets received")
    packets_read: int = Field(default=0, description="Number of packets read by receive operation")

    def get_stats(self) -> Stats:
        """Convert MemoryStats to general Stats object"""
        stats = Stats()
        stats.latency = 0.0  # Will be updated through update_execution_timeS
        stats.energy = 0.0  # Placeholder for energy consumption
        stats.area = 0.0  # Placeholder for area usage
        return stats


class Network:
    def __init__(self):
        self.routers = {}

    def register_router(self, router):
        self.routers[router.id] = router

    def get_router(self, router_id):
        return self.routers[router_id]


class Router:
    """Base memory component"""

    def __init__(self, network: Network, id: int, config: Config = None):
        self.id = id
        self.config = config or Config()
        self.network = network  # this is a reference to the network

        # Register this router with the network
        self.network.register_router(self)

        # Initialize stats
        self.stats = RouterStats()

    def run(self, env: simpy.Environment):
        """Run the router simulation"""
        self.env = env
        self.send_queue = simpy.Store(env)
        self.receive_buffers = {}
        total_routers = self.config.num_nodes * self.config.num_tiles_per_node
        for i in range(total_routers):
            if i != self.id:  # Skip creating a buffer for this router itself
                self.receive_buffers[i] = simpy.Store(env, capacity=1)

        # Start the send thread
        env.process(self._send_thread())

    def add_send_packet(self, target: int, data: NDArray[np.int32]):
        """Send a packet to the router"""
        packet = (self.id, target, data)  # packet: tuple = (source, target, data)
        self.send_queue.put(packet)
        yield self.env.timeout(self.config.core_config.dataMem_lat)

        # Update stats
        self.stats.packets_created += 1

    def _send_thread(self):
        """Thread to handle sending packets"""
        while True:
            # Get the next packet to send
            packet = yield self.send_queue.get()

            # Get the target router
            target_router = self.network.get_router(packet[1])  # packet: tuple = (source, target, data)

            # Send the packet to the target router
            yield target_router.receive_packet(packet)

            # Update stats
            self.stats.packets_sent += 1

    def receive_packet(self, packet):
        """Receive a packet from the network"""
        source, _, data = packet
        yield self.receive_buffers[source].put(data)

        # Update stats
        self.stats.packets_received += 1

    def read_packet(self, source: int):
        """Read a packet from the receive buffer"""
        data = yield self.receive_buffers[source].get()

        # Update stats
        self.stats.packets_read += 1

        return data

    def get_stats(self) -> Stats:
        return self.stats.get_stats()
