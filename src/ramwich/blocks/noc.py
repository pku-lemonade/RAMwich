import simpy
from pydantic import BaseModel, Field

from ..config import Config
from ..stats import Stats


class NetworkStats(BaseModel):
    """Statistics for DRAM operations"""

    # Universal metrics
    leakage_energy_per_cycle: float = Field(default=0.0, description="Leakage energy consumption for 1 cycle in pJ")
    area_inter: float = Field(default=0.0, description="Area for NOC internode in mm^2")
    area_intra: float = Field(default=0.0, description="Area for NOC intranode in mm^2")

    def get_stats(self) -> Stats:
        """Convert MemoryStats to general Stats object"""
        stats = Stats()

        # Map Network metrics to Stat object
        stats.increment_component_leakage_energy("NOC", self.leakage_energy_per_cycle)
        stats.increment_component_area("NOC internode", self.area_inter)
        stats.increment_component_area("NOC intranode", self.area_intra)

        return stats


class Network:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.noc_config = self.config.noc_config
        self.routers = {}
        self.queue_busy_cycles = 0
        self.is_tracking = False
        self.env = None
        self.monitor_process = None

        # Initialize stats
        self.stats = NetworkStats()
        self.stats.leakage_energy_per_cycle = (
            self.noc_config.noc_intra_pow_leak * self.config.num_tiles_per_node / self.noc_config.num_port
        )
        self.stats.area_inter = self.noc_config.noc_inter_area
        self.stats.area_intra = (
            self.noc_config.noc_intra_area * self.config.num_tiles_per_node / self.noc_config.num_port
        )

    def register_router(self, router):
        self.routers[router.id] = router

    def get_router(self, router_id):
        return self.routers[router_id]

    def start_tracking(self, env):
        """Start tracking busy queue cycles"""
        if self.is_tracking:
            return

        self.env = env
        self.is_tracking = True
        self.monitor_process = env.process(self._monitor_queues())

    def stop_tracking(self):
        """Stop tracking busy queue cycles"""
        if not self.is_tracking:
            return

        self.is_tracking = False
        if self.monitor_process:
            self.monitor_process.interrupt()

    def _monitor_queues(self):
        """Process to monitor router queues each cycle"""
        try:
            while True:
                # Check if any router has items in its send queue
                any_queue_not_empty = any(
                    router.is_running and router.send_queue and len(router.send_queue.items) > 0
                    for router in self.routers.values()
                )

                if any_queue_not_empty:
                    self.queue_busy_cycles += 1

                # Wait one cycle
                yield self.env.timeout(1)

        except simpy.Interrupt:
            pass

    def get_queue_busy_cycles(self):
        """Return the count of cycles where any router had a non-empty send queue"""
        return self.queue_busy_cycles

    def get_stats(self) -> Stats:
        """Get statistics for this Network"""
        return self.stats.get_stats()
