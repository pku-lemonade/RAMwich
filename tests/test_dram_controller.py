import numpy as np
import pytest
import simpy

from ramwich.blocks.dram_controller import DRAMController
from ramwich.blocks.memory import DRAM
from ramwich.config import TileConfig


class TestDRAMController:
    @pytest.fixture
    def setup(self):
        """Set up the environment, DRAM, and controller for testing"""
        # Create simulation environment
        env = simpy.Environment()

        # Create tile config with test values
        tile_config = TileConfig()
        tile_config.edram_lat = 1  # Short latency for testing
        tile_config.edram_size = 1024  # Small DRAM for testing

        # Create DRAM with test size
        dram = DRAM(tile_config)

        # Create controller
        controller = DRAMController(dram, tile_config)
        controller.run(env)

        return env, dram, controller

    def test_write_request(self, setup):
        """Test writing data to DRAM"""
        env, dram, controller = setup

        # Create test data
        start_addr = 0
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)

        # Track when request is complete
        done = [False]

        # Define process to submit write request
        def requester():
            # Submit write request
            event = controller.submit_write_request(0, start_addr, data)
            # Wait for completion
            yield event
            # Mark as done
            done[0] = True

        # Add process to env
        env.process(requester())

        # Run simulation until request should be complete
        env.run(until=10)

        # Check request completed
        assert done[0] is True

        # Check that data was written correctly
        assert np.array_equal(dram.cells[start_addr : start_addr + len(data)], data)

        # Check that validity was updated
        assert np.all(controller.valid[start_addr : start_addr + len(data)])

    def test_read_request_valid_data(self, setup):
        """Test reading valid data from DRAM"""
        env, dram, controller = setup

        # Write data to DRAM first
        start_addr = 0
        test_data = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        dram.write(start_addr, test_data)
        controller.valid[start_addr : start_addr + len(test_data)] = True

        # Track result of read
        result = [None]

        # Define process to submit read request
        def requester():
            # Submit read request
            event = controller.submit_read_request(0, start_addr, len(test_data), 1)
            # Wait for data
            data = yield event
            # Store result
            result[0] = data

        # Add process to env
        env.process(requester())

        # Run simulation until request should be complete
        env.run(until=10)

        # Check that correct data was read
        assert result[0] is not None
        assert np.array_equal(result[0], test_data)

    def test_read_invalid_data_wait(self, setup):
        """Test that read requests wait for data to become valid"""
        env, dram, controller = setup

        # Define addresses and test data
        start_addr = 0
        test_data = np.array([100, 200, 300], dtype=np.int32)

        # Track result and timing
        result = [None]
        completion_time = [0]

        # Define process to submit read request
        def read_requester():
            # Submit read request for data that isn't valid yet
            event = controller.submit_read_request(0, start_addr, len(test_data), 1)
            # Wait for data (this should block until data is valid)
            data = yield event
            # Record when we got the data
            completion_time[0] = env.now
            # Store result
            result[0] = data

        # Define process to write data after a delay
        def write_requester():
            # Wait before writing
            yield env.timeout(5)
            # Submit write request
            event = controller.submit_write_request(1, start_addr, test_data)
            yield event

        # Add processes to env
        env.process(read_requester())
        env.process(write_requester())

        # Run simulation
        env.run(until=20)

        # Check that read completed after write (completion_time > 5)
        assert completion_time[0] > 5

        # Check that correct data was read
        assert result[0] is not None
        assert np.array_equal(result[0], test_data)

    def test_multiple_requests_order(self, setup):
        """Test processing multiple requests in order of submission"""
        env, dram, controller = setup

        # Track completion order
        completion_order = []

        # Define process to submit multiple requests with different submission times
        def requester():
            # First write request
            yield env.timeout(1)
            write_event1 = controller.submit_write_request(0, 0, np.array([1, 2, 3], dtype=np.int32))

            # Second write request
            yield env.timeout(1)
            write_event2 = controller.submit_write_request(0, 10, np.array([4, 5, 6], dtype=np.int32))

            # Wait for first write to complete
            yield write_event1
            completion_order.append("write1")

            # Wait for second write to complete
            yield write_event2
            completion_order.append("write2")

            # Submit read request
            read_event = controller.submit_read_request(0, 0, 3, 1)

            # Wait for read to complete
            yield read_event
            completion_order.append("read")

        # Add process to env
        env.process(requester())

        # Run simulation
        env.run(until=30)

        # Check completion order matches submission order
        assert completion_order == ["write1", "write2", "read"]

    def test_read_after_partial_write(self, setup):
        """Test reading data after a write that partially covers the requested range"""
        env, dram, controller = setup

        # Define data and addresses
        start_addr = 0
        read_length = 10

        # First write only covers part of the read range
        partial_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)

        # Track results
        read_completed = [False]

        # Define process for test sequence
        def test_sequence():
            # First write partial data
            write_event = controller.submit_write_request(0, start_addr, partial_data)
            yield write_event

            # Try to read entire range (should wait in pending reads)
            read_event = controller.submit_read_request(0, start_addr, read_length, 1)

            # Submit second write that completes the range
            remaining_data = np.array([6, 7, 8, 9, 10], dtype=np.int32)
            write_event2 = controller.submit_write_request(0, start_addr + len(partial_data), remaining_data)
            yield write_event2

            # Now read should complete
            yield read_event
            read_completed[0] = True

        # Add process to env
        env.process(test_sequence())

        # Run simulation
        env.run(until=30)

        # Check read completed after second write made data valid
        assert read_completed[0] is True

        # Check validity of entire range
        assert np.all(controller.valid[start_addr : start_addr + read_length])
