from .config import NOCConfig
from typing import Dict, Any, Optional
from pydantic import Field, BaseModel
from .stats import Stat

class NOCStats(BaseModel):
    """Statistics tracking for Network-on-Chip (NoC) components"""

    # NoC-specific metrics
    packets_sent: int = Field(default=0, description="Total number of packets sent")
    packets_received: int = Field(default=0, description="Total number of packets received")
    flits_transmitted: int = Field(default=0, description="Total number of flits transmitted")
    bytes_transmitted: int = Field(default=0, description="Total bytes transmitted across the NoC")
    intra_node_transfers: int = Field(default=0, description="Number of intra-node data transfers")
    inter_node_transfers: int = Field(default=0, description="Number of inter-node data transfers")
    contention_events: int = Field(default=0, description="Number of contention events in the NoC")

    # Performance metrics
    active_cycles: int = Field(default=0, description="Number of active cycles")
    total_latency: int = Field(default=0, description="Total latency cycles")
    energy_consumption: float = Field(default=0.0, description="Energy consumption in mJ")

    # Track source-destination pairs for topology analysis
    _source_dest_pairs: Dict[str, int] = Field(default_factory=dict, exclude=True)
    op_counts: Dict[str, int] = Field(default_factory=dict, description="Operation counts by type")

    def record_packet_transmission(self, source: int, destination: int, packet_size_bytes: int,
                                  num_flits: int, is_intra_node: bool, contention: bool = False):
        """Record a packet transmission through the NoC"""
        self.packets_sent += 1
        self.bytes_transmitted += packet_size_bytes
        self.flits_transmitted += num_flits

        # Track intra vs inter node communication
        if is_intra_node:
            self.intra_node_transfers += 1
        else:
            self.inter_node_transfers += 1

        # Track contention
        if contention:
            self.contention_events += 1

        # Track source-destination pair
        pair_key = f"{source}->{destination}"
        if pair_key in self._source_dest_pairs:
            self._source_dest_pairs[pair_key] += 1
        else:
            self._source_dest_pairs[pair_key] = 1

    def record_packet_reception(self):
        """Record a packet reception"""
        self.packets_received += 1

    def get_stats(self, noc_id: Optional[int] = None, include_paths: bool = False) -> Stat:
        """Get NoC-specific statistics"""
        # Create a new Stat object
        stats = Stat()

        # Map NoC metrics to Stat object
        stats.latency = float(self.total_latency)
        stats.energy = float(self.energy_consumption)
        stats.area = 0.0  # NoC area not tracked in this component

        # Map operation counts
        stats.operations = self.packets_sent + self.packets_received
        stats.op_counts = self.op_counts.copy()

        # We could map specific operation types if needed
        # For now, we'll leave the specific operation counts at 0

        # Set execution time metrics
        stats.total_execution_time = float(self.active_cycles)

        return stats

class NetworkOnChip:
    """Hardware implementation of the Network-on-Chip component"""

    def __init__(self, noc_config=None, packet_width=16, flit_width=32, noc_id=0):
        self.noc_config = noc_config if noc_config else NOCConfig()
        self.noc_id = noc_id

        # Topology properties
        self.cmesh_c = 4  # Number of concentrations
        self.flit_width = flit_width
        self.packet_width = packet_width

        # Initialize stats
        self.stats = NOCStats()

    def send_packet(self, source_tile, dest_tile, data_size):
        """Simulate sending a packet through the NoC"""
        # Determine if this is intra-node or inter-node communication
        is_intra_node = (source_tile // self.cmesh_c == dest_tile // self.cmesh_c)

        if is_intra_node:
            latency = self.noc_config.intra_lat
            power = self.noc_config.intra_pow_dyn
        else:
            latency = self.noc_config.inter_lat
            power = self.noc_config.inter_pow_dyn

        # Calculate number of packets needed for this data
        num_packets = (data_size + self.packet_width - 1) // self.packet_width

        # Calculate flits per packet (assume header + data)
        flits_per_packet = 1 + (self.packet_width + self.flit_width - 1) // self.flit_width
        total_flits = flits_per_packet * num_packets

        # Update stats for each packet
        for _ in range(num_packets):
            self.stats.record_packet_transmission(
                source=source_tile,
                destination=dest_tile,
                packet_size_bytes=self.packet_width,
                num_flits=flits_per_packet,
                is_intra_node=is_intra_node,
                contention=False  # In a more detailed simulator, detect actual contention
            )
            self.stats.record_packet_reception()

        # Update performance stats
        self.stats.active_cycles += latency
        self.stats.total_latency += latency * num_packets
        self.stats.energy_consumption += power * latency * num_packets / 1000  # Convert from μW to mJ

        return latency * num_packets

    def get_average_latency(self):
        """Return the average packet latency"""
        if self.stats.packets_sent == 0:
            return 0
        return self.stats.total_latency / self.stats.packets_sent

    def get_energy_consumption(self):
        """Return the total energy consumption in mJ"""
        return self.stats.energy_consumption

    def get_stats(self, include_paths=False) -> Stat:
        """Return detailed statistics about this NoC"""
        return self.stats.get_stats(self.noc_id, include_paths)
