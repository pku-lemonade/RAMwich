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


class CoreExecutionTimingVisitor(CommonVisitor):
    """Visitor for calculating operation execution timing"""

    def __init__(self, config):
        self.config = config

    def visit_load(self, op):
        return self.config.load_execution_time

    def visit_store(self, op):
        return self.config.load_execution_time  # Assuming store uses load time

    def visit_set(self, op):
        return self.config.set_execution_time

    def visit_copy(self, op):
        return self.config.set_execution_time  # Assuming copy uses set time

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
            value = self.core.dram.read(op.read)
            self.core.sram.write(op.dest, value)
            self.core.stats.increment_op_count("load")
        except IndexError as e:
            logger.error(f"Load operation failed: {e}")

    def visit_store(self, op):
        try:
            value = self.core.sram.read(op.read)
            self.core.dram.write(op.dest, value)
            self.core.stats.increment_op_count("store")
        except IndexError as e:
            logger.error(f"Store operation failed: {e}")

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

    def __init__(self, core, config):
        self.functional_visitor = CoreExecutionFunctionalVisitor(core)
        self.timing_visitor = CoreExecutionTimingVisitor(config)

    def _visit_common(self, op):
        op.accept(self.functional_visitor)
        time = op.accept(self.timing_visitor)
        return time
