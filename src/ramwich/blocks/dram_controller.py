from typing import Optional

import numpy as np
import simpy
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from ..config import TileConfig
from ..stats import Stats, StatsDict
from .memory import DRAM


class Request(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    core_id: int
    start: int
    submit_time: float
    done_event: simpy.Event

    @property
    def length(self) -> Optional[int]:
        """Operride in subclasses"""
        return None


class ReadRequest(Request):
    batch_size: int
    num_batches: int

    @property
    def length(self) -> int:
        """Calculate the length of the read request"""
        return self.batch_size * self.num_batches


class WriteRequest(Request):
    data: NDArray[np.int32]

    @property
    def length(self) -> int:
        """Calculate the length of the read request"""
        return self.data.size

    @property
    def num_batches(self) -> int:
        """Calculate the batch size of the read request"""
        return self.data.shape[0]


class DRAMControllerStats(BaseModel):
    """Statistics for DRAM controller operations"""

    # Universal metrics
    config: TileConfig = Field(default=TileConfig(), description="Tile configuration")

    # DRAM controller specific metrics
    read_requests: int = Field(default=0, description="Number of read requests")
    write_requests: int = Field(default=0, description="Number of write requests")
    total_requests: int = Field(default=0, description="Total number of requests")
    total_wait_time: int = Field(default=0.0, description="Total wait time for requests")
    max_wait_time: int = Field(default=0.0, description="Maximum wait time for requests")
    active_cycles: int = Field(default=0, description="Number of active cycles")
    operating_time: int = Field(default=0, description="Total operating time")

    def reset(self):
        """Reset the statistics"""
        self.read_requests = 0
        self.write_requests = 0
        self.total_requests = 0
        self.total_wait_time = 0
        self.max_wait_time = 0
        self.active_cycles = 0
        self.operating_time = 0

    def get_stats(self) -> StatsDict:
        """Convert DRAMControllerStats to general Stats object"""
        stats = Stats(
            activation_count=self.total_requests,
            dynamic_energy=self.config.edram_ctrl_pow_dyn * self.total_requests,
            leakage_energy=self.config.edram_ctrl_pow_leak + self.config.edram_bus_pow_leak,
            area=self.config.edram_ctrl_area + self.config.edram_bus_area,
        )

        return StatsDict({"DRAM Controller": stats})


class DRAMController:
    def __init__(self, dram: DRAM, tile_config: TileConfig):
        # Configuration
        self.dram = dram
        self.tile_config = tile_config
        self.is_running = False

        # Validity array for the DRAM
        self.valid = np.zeros(dram.size, dtype=np.bool_)

        # Placeholders for simpy objects, will be initialized in run()
        self.env = None  # Environment
        self.memory_bus = None  # Memory bus resource for exclusive access to DRAM
        self.requests = None  # Request queue

        # Request queue for reads that is not valid yet
        self.pending_reads = []

        # Initialize stats
        self.stats = DRAMControllerStats(config=self.tile_config)

    def run(self, env: simpy.Environment):
        """Initialize the DRAM controller with the simulation environment"""
        if self.is_running:
            return

        self.env = env
        self.memory_bus = simpy.Resource(env, capacity=1)
        self.requests = simpy.Store(env)
        self.is_running = True
        self.start_time = env.now

        # Start the request handler process
        self.handler_process = self.env.process(self.request_handler())

    def stop(self):
        """Stop the DRAM controller"""
        if not self.is_running:
            return

        self.is_running = False

        # Calculate active time
        active_time = self.env.now - self.start_time

        # Interrupt the handler process if it's still running
        if hasattr(self, "handler_process") and not self.handler_process.triggered:
            self.handler_process.interrupt()

        # Process any remaining pending reads
        if self.pending_reads:
            for request in self.pending_reads[:]:
                # Fail any pending read requests that couldn't be fulfilled
                if not request.done_event.triggered:
                    request.done_event.fail(RuntimeError("DRAM controller stopped before request could be fulfilled"))

        # Update stats with active time
        self.stats.active_cycles += int(active_time)

    def submit_read_request(self, core_id: int, start: int, batch_size: int, num_batches: int) -> simpy.Event:
        """Submit a read request to the DRAM controller"""

        if not self.is_running:
            raise RuntimeError("DRAM controller not running. Call run() first.")

        # Create an event that will be triggered when request completes
        done_event = self.env.event()

        # Add request to queue with completion event
        request = ReadRequest(
            core_id=core_id,
            start=start,
            batch_size=batch_size,
            num_batches=num_batches,
            submit_time=self.env.now,
            done_event=done_event,
        )

        # Add the request to the ready requests
        self.requests.put(request)

        # Update stats
        self.stats.read_requests += num_batches
        self.stats.total_requests += num_batches

        # Return event that the core can yield on
        return done_event

    def submit_write_request(self, core_id: int, start: int, data: NDArray[np.int32]) -> simpy.Event:
        """Submit a write request to the DRAM controller"""

        if not self.is_running:
            raise RuntimeError("DRAM controller not running. Call run() first.")

        # Create an event that will be triggered when request completes
        done_event = self.env.event()

        # Add request to queue with completion event
        request = WriteRequest(core_id=core_id, start=start, data=data, submit_time=self.env.now, done_event=done_event)

        # Add the request to the ready requests
        self.requests.put(request)

        # Update stats
        self.stats.write_requests += request.num_batches
        self.stats.total_requests += request.num_batches

        # Return event that the core can yield on
        return done_event

    def request_handler(self):
        """Process that continuously handles memory requests"""
        try:
            while True:
                # Get the next request
                request = yield self.requests.get()

                # Depending on the type of request, call the appropriate handler
                if isinstance(request, ReadRequest):
                    # first check if the data is valid
                    if np.all(self.valid[request.start : request.start + request.length]):
                        # If the data is valid, process the read request
                        self.env.process(self._read_thread(request))
                    else:
                        # If the data is not valid, add it to the pending reads
                        self.pending_reads.append(request)

                elif isinstance(request, WriteRequest):
                    # Handle write request
                    self.env.process(self._write_thread(request))
        except simpy.Interrupt:
            pass

    def _read_thread(self, request: ReadRequest):
        """Thread to handle read requests"""
        # Request exclusive access to the memory bus
        with self.memory_bus.request() as req:
            # Wait for the bus to be available
            yield req

            # Calculate read latency based on number of batches
            latency = self.tile_config.edram_lat * request.num_batches

            # Simulate the time it takes to read
            yield self.env.timeout(latency)

            # Perform the actual read
            data = self.dram.read(request.start, request.length, batch=request.num_batches)

            # Update stats
            self.stats.total_wait_time += self.env.now - request.submit_time
            self.stats.max_wait_time = max(self.stats.max_wait_time, self.env.now - request.submit_time)
            self.stats.operating_time += int(latency)

            # Notify waiting core with result
            request.done_event.succeed(data)

    def _write_thread(self, request: WriteRequest):
        """Thread to handle write requests"""
        # Request exclusive access to the memory bus
        with self.memory_bus.request() as req:
            # Wait for the bus to be available
            yield req

            # Calculate write latency based on number of rows
            latency = self.tile_config.edram_lat * request.num_batches

            # Simulate the time it takes to write
            yield self.env.timeout(latency)

            # Perform the actual write
            data = request.data.flatten()
            self.dram.write(request.start, data, batch=request.num_batches)

            # Update the validity array
            self.valid[request.start : request.start + request.length] = True

            # Update the ready requests
            self._update_ready_requests()

            # Update stats
            self.stats.total_wait_time += self.env.now - request.submit_time
            self.stats.max_wait_time = max(self.stats.max_wait_time, self.env.now - request.submit_time)
            self.stats.operating_time += int(latency)

            # Notify waiting core that write is complete
            request.done_event.succeed()

    def _update_ready_requests(self):
        """Update the ready requests in the queue"""
        # Check if there are any pending reads
        if self.pending_reads:
            # If there are pending reads, add them to the ready requests
            for request in self.pending_reads:
                # If the data is valid, add it to the ready requests
                if np.all(self.valid[request.start : request.start + request.length]):
                    self.requests.put(request)
                    self.pending_reads.remove(request)

    def reset(self):
        """Reset the DRAM controller and its statistics"""
        self.dram.reset()
        self.valid.fill(False)
        self.stats.reset()

    def get_stats(self):
        return self.stats.get_stats()
