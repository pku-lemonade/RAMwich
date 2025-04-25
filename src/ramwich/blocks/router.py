import numpy as np
import simpy
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import Config
from ..stats import Stats


class RouterStats(BaseModel):
    """Statistics for DRAM operations"""

    # Universal metrics
    unit_energy_consumption_receive: float = Field(
        default=0.0, description="Energy consumption for each packet received in pJ"
    )
    unit_energy_consumption_send_inter: float = Field(
        default=0.0, description="Energy consumption for each packet sent inter_node in pJ"
    )
    unit_energy_consumption_send_intra: float = Field(
        default=0.0, description="Energy consumption for each packet sent intra_node in pJ"
    )
    leakage_energy_per_cycle: float = Field(default=0.0, description="Leakage energy consumption for 1 cycle in pJ")
    area: float = Field(default=0.0, description="Area in mm^2")

    # Router specific metrics
    packets_created: int = Field(default=0, description="Number of packets created")
    packets_sent_internode: int = Field(default=0, description="Number of packets sent to other nodes")
    packets_sent_intranode: int = Field(default=0, description="Number of packets sent to the same node")
    packets_sent: int = Field(default=0, description="Number of packets sent")
    packets_received: int = Field(default=0, description="Number of packets received")
    packets_read: int = Field(default=0, description="Number of packets read by receive operation")

    def get_stats(self) -> Stats:
        """Convert MemoryStats to general Stats object"""
        stats = Stats()

        # Map Router metrics to Stat object
        stats.increment_component_activation("Router send internode", self.packets_sent_internode)
        stats.increment_component_activation("Router send intranode", self.packets_sent_intranode)
        stats.increment_component_activation("Router receive", self.packets_received)
        stats.increment_component_dynamic_energy(
            "Router send internode", self.unit_energy_consumption_send_inter * self.packets_sent_internode
        )
        stats.increment_component_dynamic_energy(
            "Router send intranode", self.unit_energy_consumption_send_intra * self.packets_sent_intranode
        )
        stats.increment_component_dynamic_energy(
            "Router receive", self.unit_energy_consumption_receive * self.packets_received
        )
        stats.increment_component_leakage_energy("Router", self.leakage_energy_per_cycle)
        stats.increment_component_area("Router", self.area)

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
        self.is_running = False

        # Register this router with the network
        self.network.register_router(self)

        # Initialize stats
        self.stats = RouterStats()
        self.stats.unit_energy_consumption_receive = self.config.tile_config.receive_buffer_pow_dyn
        self.stats.unit_energy_consumption_send_inter = self.config.noc_config.noc_inter_pow_dyn
        self.stats.unit_energy_consumption_send_intra = self.config.noc_config.noc_intra_pow_dyn
        self.stats.leakage_energy_per_cycle = self.config.tile_config.receive_buffer_pow_leak
        self.stats.area = self.config.tile_config.receive_buffer_area

    def run(self, env: simpy.Environment):
        """Run the router simulation"""
        if self.is_running:
            return

        self.env = env
        self.send_queue = simpy.Store(env)
        self.receive_buffers = {}
        total_routers = self.config.num_nodes * self.config.num_tiles_per_node
        for i in range(total_routers):
            if i != self.id:  # Skip creating a buffer for this router itself
                self.receive_buffers[i] = simpy.Store(env, capacity=1)

        # Start the send thread
        self.sending_event = None
        self.send_process = env.process(self._send_thread())

        # Set the router to running state
        self.is_running = True

    def stop(self):
        """Stop the router"""

        if not self.is_running:
            return

        self.is_running = False
        self.send_process.interrupt()  # Stop the send process

    def stop_after_all_packets_sent(self):
        """Stop the router after all packets have been sent"""
        if not self.is_running:
            return

        # Create a specific event that will trigger when send queue empties
        empty_event = self.env.event()

        def check_queue():
            if self.send_queue.items:
                # Still items in queue, check again after some time
                yield self.env.timeout(1)
                self.env.process(check_queue())
            else:
                # Queue is empty, trigger the event
                empty_event.succeed()

        # Start the checking process
        self.env.process(check_queue())

        # Wait for the send queue to be empty
        yield empty_event

        # Wait for the sending event to finish
        if self.sending_event:
            yield self.sending_event

        # Stop the send process
        self.stop()

    def _get_latency(self, source: int, target: int):
        """Calculate the latency between two routers"""
        # input and output tiles are considered to be on a different node with other tiles
        if (
            source == 0
            or target == 1
            or source // self.config.num_tiles_per_node != target // self.config.num_tiles_per_node
        ):
            # Inter-node communication
            # Also return a flag indicating that this is inter-node communication
            return self.config.noc_config.noc_intra_lat + self.config.noc_config.noc_inter_lat, True
        else:
            # Intra-node communication
            return self.config.noc_config.noc_intra_lat, False

    def add_send_packet(self, target: int, data: NDArray[np.int32]):
        """Send a packet to the router"""
        packet = (self.id, target, data)  # packet: tuple = (source, target, data)
        self.send_queue.put(packet)
        yield self.env.timeout(self.config.core_config.dataMem_lat)

        # Update stats
        self.stats.packets_created += 1

    def _send_thread(self):
        """Thread to handle sending packets"""
        try:
            while True:
                # Get the next packet to send
                packet = yield self.send_queue.get()

                # Once a packet is get, create a new sending event
                self.sending_event = self.env.event()

                # Get the target router
                target_router = self.network.get_router(packet[1])  # packet: tuple = (source, target, data)

                # Simulate the time taken to send the packet
                latency, is_inter_node = self._get_latency(packet[0], packet[1])
                yield self.env.timeout(latency)

                # Send the packet to the target router
                yield self.env.process(target_router.receive_packet(packet))

                # Mark the send event as done
                self.sending_event.succeed()

                # Update stats
                self.stats.packets_sent += 1
                if is_inter_node:
                    self.stats.packets_sent_internode += 1
                else:
                    self.stats.packets_sent_intranode += 1
        except simpy.Interrupt:
            pass

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
