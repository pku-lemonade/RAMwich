from .config import NOCConfig

class NetworkOnChip:
    """Hardware implementation of the Network-on-Chip component"""

    def __init__(self, noc_config=None, packet_width=16, flit_width=32):
        self.noc_config = noc_config if noc_config else NOCConfig()

        # Topology properties
        self.cmesh_c = 4  # Number of concentrations
        self.flit_width = flit_width
        self.packet_width = packet_width

        # Performance tracking
        self.packet_count = 0
        self.total_latency = 0
        self.active_cycles = 0
        self.energy_consumption = 0

    def send_packet(self, source_tile, dest_tile, data_size):
        """Simulate sending a packet through the NoC"""
        # Determine if this is intra-node or inter-node communication
        if source_tile // self.cmesh_c == dest_tile // self.cmesh_c:
            latency = self.noc_config.intra_lat
            power = self.noc_config.intra_pow_dyn
        else:
            latency = self.noc_config.inter_lat
            power = self.noc_config.inter_pow_dyn

        # Calculate number of packets needed for this data
        num_packets = (data_size + self.packet_width - 1) // self.packet_width

        # Update stats
        self.packet_count += num_packets
        self.total_latency += latency * num_packets
        self.active_cycles += latency
        self.energy_consumption += power * latency * num_packets

        return latency * num_packets

    def get_average_latency(self):
        """Return the average packet latency"""
        if self.packet_count == 0:
            return 0
        return self.total_latency / self.packet_count

    def get_energy_consumption(self):
        """Return the total energy consumption in mJ"""
        return self.energy_consumption / 1000  # Convert from μW to mJ
