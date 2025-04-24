import logging
from abc import ABC, abstractmethod

import numpy as np

from .ops import MVM, VFU, Copy, Hlt, Load, Set, Store

logger = logging.getLogger(__name__)


class CoreVisitor(ABC):
    """Abstract base class for core operation visitors"""

    @abstractmethod
    def visit_load(self, op: Load):
        pass

    @abstractmethod
    def visit_store(self, op: Store):
        pass

    @abstractmethod
    def visit_set(self, op: Set):
        pass

    @abstractmethod
    def visit_copy(self, op: Copy):
        pass

    @abstractmethod
    def visit_mvm(self, op: MVM):
        pass

    @abstractmethod
    def visit_vfu(self, op: VFU):
        pass

    @abstractmethod
    def visit_hlt(self, op: Hlt):
        pass


class CommonVisitor(CoreVisitor):
    """Abstract base class for visitors where all visit methods perform a common action."""

    @abstractmethod
    def _visit_common(self, op):
        """Common action performed by all visit methods."""
        pass

    def visit_load(self, op: Load):
        return self._visit_common(op)

    def visit_store(self, op: Store):
        return self._visit_common(op)

    def visit_set(self, op: Set):
        return self._visit_common(op)

    def visit_copy(self, op: Copy):
        return self._visit_common(op)

    def visit_mvm(self, op: MVM):
        return self._visit_common(op)

    def visit_vfu(self, op: VFU):
        return self._visit_common(op)

    def visit_hlt(self, op: Hlt):
        return self._visit_common(op)


class CoreFetchVisitor(CommonVisitor):
    """Visitor for calculating fetch timing"""

    def __init__(self, core):
        self.core = core
        self.fetch_time = core.config.core_config.instrnMem_lat

    def _visit_common(self, op):
        return self.core.env.timeout(self.fetch_time)


class CoreDecodeVisitor(CommonVisitor):
    """Visitor for calculating decode timing"""

    def __init__(self, core):
        self.core = core
        self.decode_time = core.config.core_config.instrnMem_lat

    def _visit_common(self, op):
        return self.core.env.timeout(self.decode_time)


class CoreExecutionTimingVisitor(CoreVisitor):
    """Visitor for calculating operation execution timing"""

    def __init__(self, config):
        self.config = config

    def visit_load(self, op):
        # This should not be used for load operations
        raise NotImplementedError("Load operations should not use this visitor")

    def visit_store(self, op):
        # This should not be used for store operations
        raise NotImplementedError("Store operations should not use this visitor")

    def visit_set(self, op):
        """Calculate set execution time"""
        return self.config.core_config.dataMem_lat

    def visit_copy(self, op):
        """Calculate copy execution time"""
        return self.config.core_config.dataMem_lat

    def visit_vfu(self, op):
        """Calculate VFU execution time"""
        return (
            self.config.core_config.alu_lat
            * (op.vec + self.config.core_config.num_alu_per_vfu - 1)
            // self.config.core_config.num_alu_per_vfu
        )

    def visit_mvm(self, op):
        """Calculate MVM execution time"""
        # This is now synchronized with PUMA. Needs to be recalculated
        return self.config.mvmu_config.adc_config.lat * (
            (self.config.data_width + self.config.mvmu_config.dac_config.resolution - 1)
            // self.config.mvmu_config.dac_config.resolution
            + 2
        )

    def visit_hlt(self, op):
        return 1  # Minimal time unit for halt


class CoreExecutionVisitor(CoreVisitor):
    """Visitor for executing operations functionally"""

    def __init__(self, core):
        self.core = core
        self.timing_visitor = CoreExecutionTimingVisitor(core.config)

    def visit_load(self, op):
        # Create an event to signal when the load operation is complete
        done_event = self.core.env.event()

        # First read the DRAM address from the register
        dram_address = self.core.read_from_register(op.read, 1)
        dram_address = dram_address[0]  # return value from read is a vector

        # Send read request to DRAM controller
        read_event = self.core.dram_controller.submit_read_request(
            core_id=self.core.id,
            start=dram_address,
            batch_size=op.width,
            num_batches=op.vec,
        )

        # Create callback function to handle result when data arrives
        def on_dram_read_complete(event):
            try:
                # Write to register when data is available
                data = read_event.value  # Get data from event

                def complete_write_after_latency():
                    # Simulate a delay for the write to register
                    latency = self.core.core_config.dataMem_lat * op.vec
                    yield self.core.env.timeout(latency)

                    self.core.write_to_register(op.dest, data)
                    done_event.succeed()

                self.core.env.process(complete_write_after_latency())
            except Exception as e:
                logger.error(f"Load completion failed: {e}")
                done_event.fail(e)

        # Schedule callback when read completes
        read_event.callbacks.append(on_dram_read_complete)

        # Update operation count
        self.core.stats.increment_op_count("load")

        # Return the done event to the caller
        return done_event

    def visit_store(self, op):
        # First read the DRAM address from the register
        dram_address = self.core.read_from_register(op.dest, 1)
        dram_address = dram_address[0]  # return value from read is a vector

        # Read data from the register to be stored and reshape it
        data = self.core.read_from_register(op.read, op.width * op.vec)
        data = np.reshape(data, (op.vec, op.width))

        def send_write_request_after_latency():
            # Simulate a delay for the read from register
            latency = self.core.core_config.dataMem_lat * op.vec
            yield self.core.env.timeout(latency)

            # Send write request to DRAM controller
            write_event = self.core.dram_controller.submit_write_request(
                core_id=self.core.id,
                start=dram_address,
                data=data,
            )

            yield write_event

        # Schedule the write request after the read completes
        write_event = self.core.env.process(send_write_request_after_latency())

        # Update operation count
        self.core.stats.increment_op_count("store")

        # Return the done event to the caller
        return write_event

    def visit_set(self, op):
        # create a vector of size vec with the immediate value
        vector = np.full(op.vec, op.imm)
        # write the vector to the destination address
        self.core.write_to_register(op.dest, vector)

        # Update operation count
        self.core.stats.increment_op_count("set")

        # return the done event to the caller
        # done_event is a timeout event since this operation takes fixed time
        return self.core.env.timeout(op.accept(self.timing_visitor))

    def visit_copy(self, op):
        vector = self.core.read_from_register(op.read, op.vec)
        self.core.write_to_register(op.dest, vector)

        # Update operation count
        self.core.stats.increment_op_count("copy")

        # return the done event to the caller
        # done_event is a timeout event since this operation takes fixed time
        return self.core.env.timeout(op.accept(self.timing_visitor))

    def visit_vfu(self, op):
        a = self.core.read_from_register(op.read_1, op.vec)
        if op.read_2 is not None:
            b = self.core.read_from_register(op.read_2, op.vec)
            result = self.core.vfu.calculate(op.opcode, a, b)
        else:
            result = self.core.vfu.calculate(op.opcode, a)
        self.core.write_to_register(op.dest, result)

        # Update operation count
        self.core.stats.increment_op_count("vfu")

        # return the done event to the caller
        # done_event is a timeout event since this operation takes fixed time
        return self.core.env.timeout(op.accept(self.timing_visitor))

    def visit_mvm(self, op):
        for mvmu_id in op.xbar:
            self.core.get_mvmu(mvmu_id).execute_mvm()

        # Update operation count
        self.core.stats.increment_op_count("mvm")

        # return the done event to the caller
        # done_event is a timeout event since this operation takes fixed time
        return self.core.env.timeout(op.accept(self.timing_visitor))

    def visit_hlt(self, op):
        # Update operation count
        self.core.stats.increment_op_count("hlt")

        # return the done event to the caller
        # done_event is a timeout event since this operation takes fixed time
        return self.core.env.timeout(op.accept(self.timing_visitor))
