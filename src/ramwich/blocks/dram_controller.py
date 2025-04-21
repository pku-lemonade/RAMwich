from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import simpy
from numpy.typing import NDArray

from ..config import TileConfig
from .memory import DRAM


class RequestType(Enum):
    READ = 0
    WRITE = 1


class DRAMController:
    def __init__(self, env: simpy.Environment, dram: DRAM, tile_config: TileConfig = None):
        # Configuration
        self.env = env
        self.dram = dram
        self.tile_config = tile_config or TileConfig()

        # Resource that enforces exclusive access to DRAM
        self.memory_bus = simpy.Resource(env, capacity=1)

        # Request queue statistics
        self.waiting_requests = 0
        self.total_wait_time = 0
        self.request_count = 0

        # Start the request handler process
        self.env.process(self.request_handler())

        # Validity array for the DRAM
        self.valid = np.zeros(dram.size, dtype=np.bool_)

        # Queue for pending requests (core_id, request_type, address, length/data)
        self.request_queue = []

    def submit_read_request(self, core_id: int, start: int, batch_size: int, num_batches: int) -> simpy.Event:
        """Submit a read request to the DRAM controller"""
        # Create an event that will be triggered when request completes
        done_event = self.env.event()

        # Add request to queue with completion event
        self.request_queue.append(
            {
                "core_id": core_id,
                "type": RequestType.READ,
                "start": start,
                "batch_size": batch_size,
                "num_batches": num_batches,
                "submit_time": self.env.now,
                "done_event": done_event,
            }
        )

        self.waiting_requests += 1
        self.request_count += 1

        # Return event that the core can yield on
        return done_event

    def submit_write_request(self, core_id: int, start: int, data: NDArray[np.int32]) -> simpy.Event:
        """Submit a write request to the DRAM controller"""
        # Create an event that will be triggered when request completes
        done_event = self.env.event()

        # Add request to queue with completion event
        self.request_queue.append(
            {
                "core_id": core_id,
                "type": RequestType.WRITE,
                "start": start,
                "data": data,
                "submit_time": self.env.now,
                "done_event": done_event,
            }
        )

        self.waiting_requests += 1
        self.request_count += 1

        # Return event that the core can yield on
        return done_event

    def request_handler(self):
        """Process that continuously handles memory requests"""
        while True:
            if self.request_queue:
                # Get the next request
                request = self.request_queue.pop(0)
                self.waiting_requests -= 1

                # Calculate wait time for statistics
                wait_time = self.env.now - request["submit_time"]
                self.total_wait_time += wait_time

                # Request exclusive access to the memory bus
                with self.memory_bus.request() as req:
                    # Wait for the bus to be available
                    yield req

                    # Process the request
                    if request["type"] == RequestType.READ:
                        # Calculate the total length of the read request
                        length = request["batch_size"] * request["num_batches"]
                        start = request["start"]

                        while not np.all(self.valid[start : start + length]):
                            # Wait for the data to be valid
                            yield self.env.timeout(1)

                            # Re-request the memory bus
                            with self.memory_bus.request() as poll_req:
                                yield poll_req

                        # Calculate read latency based on number of batches
                        latency = self.tile_config.edram_latency * request["num_batches"]

                        # Simulate the time it takes to read
                        yield self.env.timeout(latency)

                        # Perform the actual read
                        data = self.dram.read(start, length)

                        # Notify waiting core with result
                        request["done_event"].succeed(data)

                    elif request["type"] == RequestType.WRITE:
                        start = request["start"]
                        data = request["data"]

                        # Get dimensions of the data matrix
                        num_rows = data.shape[0]
                        num_cols = data.shape[1]
                        total_elements = num_rows * num_cols

                        # Calculate latency based on number of rows
                        latency = self.tile_config.edram_latency * num_rows

                        # Simulate the time it takes to write
                        yield self.env.timeout(latency)

                        # Perform the actual write
                        data = data.flatten()
                        self.dram.write(start, data)

                        # Update the validity array
                        self.valid[start : start + total_elements] = True

                        # Notify waiting core that write is complete
                        request["done_event"].succeed()
            else:
                # No requests, yield control and wait for next request
                yield self.env.timeout(1)
