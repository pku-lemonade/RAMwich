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

    def __init__(self, config):
        self.fetch_time = config.fetch_execution_time

    def _visit_common(self, op):
        return self.fetch_time


class CoreDecodeVisitor(CommonVisitor):
    """Visitor for calculating decode timing"""

    def __init__(self, config):
        self.decode_time = config.decode_execution_time

    def _visit_common(self, op):
        return self.decode_time


class CoreExecutionTimingVisitor(CoreVisitor):
    """Visitor for calculating operation execution timing"""

    def __init__(self, config):
        self.config = config

    def visit_load(self, op):
        return self.config.load_execution_time

    def visit_store(self, op):
        return self.config.store_execution_time

    def visit_set(self, op):
        return self.config.set_execution_time

    def visit_copy(self, op):
        return self.config.copy_execution_time

    def visit_vfu(self, op):
        return self.config.vfu_execution_time

    def visit_mvm(self, op):
        return self.config.mvm_execution_time

    def visit_hlt(self, op):
        return 1  # Minimal time unit for halt


class CoreExecutionFunctionalVisitor(CoreVisitor):
    """Visitor for executing operations functionally"""

    def __init__(self, core):
        self.core = core

    def visit_load(self, op):
        try:
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
                    self.core.write_to_register(op.dest, data)
                    # Signal completion
                    done_event.succeed()
                except Exception as e:
                    logger.error(f"Load completion failed: {e}")
                    done_event.fail(e)

            # Schedule callback when read completes
            read_event.callbacks.append(on_dram_read_complete)

            # Update operation count
            self.core.stats.increment_op_count("load")

            # Return the done event to the caller
            return done_event

        except IndexError as e:
            logger.error(f"Load operation failed: {e}")
            # Create and fail an event on error
            fail_event = self.env.event()
            fail_event.fail(e)
            return fail_event

    def visit_store(self, op):
        try:
            # First read the DRAM address from the register
            dram_address = self.core.read_from_register(op.dest, 1)
            dram_address = dram_address[0]  # return value from read is a vector

            # Read data from the register to be stored and reshape it
            data = self.core.read_from_register(op.read, op.width * op.vec)
            data = np.reshape(data, (op.vec, op.width))

            # Send write request to DRAM controller
            write_event = self.core.dram_controller.submit_write_request(
                core_id=self.core.id,
                start=dram_address,
                data=data,
            )

            # Update operation count
            self.core.stats.increment_op_count("store")

            # Return the done event to the caller
            return write_event

        except IndexError as e:
            logger.error(f"Store operation failed: {e}")
            # Create and fail an event on error
            fail_event = self.env.event()
            fail_event.fail(e)
            return fail_event

    def visit_set(self, op):
        try:
            # create a vector of size vec with the immediate value
            vector = np.full(op.vec, op.imm)
            # write the vector to the destination address
            self.core.write_to_register(op.dest, vector)

            # Update operation count
            self.core.stats.increment_op_count("set")

        except IndexError as e:
            logger.error(f"Set operation failed: {e}")

    def visit_copy(self, op):
        try:
            vector = self.core.read_from_register(op.read, op.vec)
            self.core.write_to_register(op.dest, vector)

            # Update operation count
            self.core.stats.increment_op_count("copy")

        except IndexError as e:
            logger.error(f"Copy operation failed: {e}")

    def visit_vfu(self, op):
        try:
            a = self.core.read_from_register(op.read_1, op.vec)
            if op.read_2 is not None:
                b = self.core.read_from_register(op.read_2, op.vec)
                result = self.core.vfu.calculate(op.opcode, a, b)
            else:
                result = self.core.vfu.calculate(op.opcode, a)
            self.core.write_to_register(op.dest, result)

            # Update operation count
            self.core.stats.increment_op_count("vfu")

        except IndexError as e:
            logger.error(f"VFU operation failed: {e}")

    def visit_mvm(self, op):
        try:
            for mvmu_id in op.xbar:
                self.core.get_mvmu(mvmu_id).execute_mvm()

            # Update operation count
            self.core.stats.increment_op_count("mvm")

        except Exception as e:
            logger.error(f"MVM operation failed: {e}")

    def visit_hlt(self, op):
        pass


class CoreExecutionVisitor(CommonVisitor):
    """Visitor that performs functional execution and returns timing"""

    def __init__(self, core):
        self.functional_visitor = CoreExecutionFunctionalVisitor(core)
        self.timing_visitor = CoreExecutionTimingVisitor(core.config)

    def _visit_common(self, op):
        done_event = op.accept(self.functional_visitor)
        if done_event:
            # If a done event is returned, the stage will uses it to wait for completion
            return done_event
        else:
            # If no event is returned, the stage will use a timeout to wait for completion
            time = op.accept(self.timing_visitor)
            return time
