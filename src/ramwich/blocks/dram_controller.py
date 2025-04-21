from typing import Optional

import numpy as np
import simpy
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import TileConfig
from .memory import DRAM


class Request(BaseModel):
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


class WriteRequest(BaseModel):
    data: NDArray[np.int32]

    @property
    def length(self) -> int:
        """Calculate the length of the read request"""
        return self.data.size

    @property
    def num_batches(self) -> int:
        """Calculate the batch size of the read request"""
        return self.data.shape[0]


class DRAMController:
    def __init__(self, env: simpy.Environment, dram: DRAM, tile_config: TileConfig = None):
        # Configuration
        self.env = env
        self.dram = dram
        self.tile_config = tile_config or TileConfig()

        # Resource that enforces exclusive access to DRAM
        self.memory_bus = simpy.Resource(env, capacity=1)

        # Start the request handler process
        self.env.process(self.request_handler())

        # Validity array for the DRAM
        self.valid = np.zeros(dram.size, dtype=np.bool_)

        # Request queue
        self.requests = simpy.Store(env)
        self.pending_reads = []

    def submit_read_request(self, core_id: int, start: int, batch_size: int, num_batches: int) -> simpy.Event:
        """Submit a read request to the DRAM controller"""
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

        # Return event that the core can yield on
        return done_event

    def submit_write_request(self, core_id: int, start: int, data: NDArray[np.int32]) -> simpy.Event:
        """Submit a write request to the DRAM controller"""
        # Create an event that will be triggered when request completes
        done_event = self.env.event()

        # Add request to queue with completion event
        request = WriteRequest(core_id=core_id, start=start, data=data, submit_time=self.env.now, done_event=done_event)

        # Add the request to the ready requests
        self.requests.put(request)

        # Return event that the core can yield on
        return done_event

    def request_handler(self):
        """Process that continuously handles memory requests"""
        while True:
            # Get the next request
            request = yield self.ready_requests.get()

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
            data = self.dram.read(request.start, request.length)

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
            self.dram.write(request.start, data)

            # Update the validity array
            self.valid[request.start : request.start + request.length] = True

            # Update the ready requests
            self._update_ready_requests()

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
                    self.ready_requests.put(request)
                    self.pending_reads.remove(request)
