import logging
from abc import ABC, abstractmethod

import numpy as np

from .ops import FVFU, IVFU, MVM, Copy, Hlt, Load, Set, Store

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
    def visit_fvfu(self, op: FVFU):
        pass

    @abstractmethod
    def visit_ivfu(self, op: IVFU):
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

    def visit_fvfu(self, op: FVFU):
        return self._visit_common(op)

    def visit_ivfu(self, op: IVFU):
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
        self.decode_time = core.config.core_config.dataMem_lat

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

    def visit_fvfu(self, op):
        """Calculate FVFU execution time"""
        return self.config.core_config.fvfu_lat * (
            (op.vec + self.config.core_config.num_alu_per_ivfu - 1) // self.config.core_config.num_alu_per_ivfu
        )

    def visit_ivfu(self, op):
        """Calculate Integer VFU execution time"""
        return (
            self.config.core_config.alu_lat
            * (op.vec + self.config.core_config.num_alu_per_ivfu - 1)
            // self.config.core_config.num_alu_per_ivfu
        )

    def visit_mvm(self, op: MVM) -> int:
        """Calculate MVM execution time"""

        mvmu   = self.config.mvmu_config
        xbar = self.config.xbar_config
        dac  = self.config.dac_config
        adc  = self.config.adc_config

        num_iter = mvmu.num_iterations
        activation_width = self.config.data_config.activation_width


        # if RRAM, use adc to read out
        if mvmu.have_rram_xbar:
            t_rram_core     = dac.lat + xbar.xbar_lat + mvmu.snh_lat
            t_rram_readout  = mvmu.mux_lat + adc.LAT_DICT[8]
        else:
            t_rram_core    = 0
            t_rram_readout = 0

        # if SRAM, consider linear and expw paths
        has_linear = len(mvmu.sram_mvm_indices) > 0 and mvmu.have_sram_xbar
        t_linear_core = (xbar.sram_xbar_lat + xbar.calculator_lat + xbar.mac_lat) if has_linear else 0

        has_expw = len(mvmu.sram_ewmvm_indices) > 0 and mvmu.have_sram_xbar
        t_expw_core = (xbar.sram_xbar_lat + xbar.calculator_lat + xbar.mac_lat + mvmu.sna_lat) if has_expw else 0

        # EAA finalize time (only once at the end)
        has_eaa = len(mvmu.sram_eaa_indices) > 0 and mvmu.have_sram_xbar
        t_eaa_finalize = mvmu.sna_lat if has_eaa else 0

        # Parallel mvmu core take the maximum of the three + RRAM
        t_core = max(t_rram_core, t_linear_core, t_expw_core)

        t_linear_accumulate = mvmu.sna_lat if has_linear else 0
        t_expw_accumulate   = mvmu.sna_lat if has_expw else 0

        per_iter = t_core + t_rram_readout + t_linear_accumulate + t_expw_accumulate

        total_cycles = (num_iter * per_iter + t_eaa_finalize) * activation_width + 2
        return total_cycles
    

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

        # Return the done event to the caller
        return write_event

    def visit_set(self, op):
        # create a vector of size vec with the immediate value
        vector = np.full(op.vec, op.imm)
        # write the vector to the destination address
        self.core.write_to_register(op.dest, vector)

        # return the done event to the caller
        # done_event is a timeout event since this operation takes fixed time
        return self.core.env.timeout(op.accept(self.timing_visitor))

    def visit_copy(self, op):
        vector = self.core.read_from_register(op.read, op.vec)
        self.core.write_to_register(op.dest, vector)

        # return the done event to the caller
        # done_event is a timeout event since this operation takes fixed time
        return self.core.env.timeout(op.accept(self.timing_visitor))

    def visit_fvfu(self, op):
        a = self.core.read_from_register(op.read_1, op.vec)
        if op.read_2 is not None:
            b = self.core.read_from_register(op.read_2, op.vec)
            result = self.core.fvfu.calculate(op.opcode, a, b, op.imm)
        else:
            result = self.core.fvfu.calculate(op.opcode, a, imm=op.imm)
        self.core.write_to_register(op.dest, result)

        # return the done event to the caller
        # done_event is a timeout event since this operation takes fixed time
        return self.core.env.timeout(op.accept(self.timing_visitor))

    def visit_ivfu(self, op):
        a = self.core.read_from_register(op.read_1, op.vec)
        if op.read_2 is not None:
            b = self.core.read_from_register(op.read_2, op.vec)
            result = self.core.ivfu.calculate(op.opcode, a, b)
        else:
            result = self.core.ivfu.calculate(op.opcode, a)
        self.core.write_to_register(op.dest, result)

        # return the done event to the caller
        # done_event is a timeout event since this operation takes fixed time
        return self.core.env.timeout(op.accept(self.timing_visitor))

    def visit_mvm(self, op):
        for mvmu_id in op.xbar:
            self.core.get_mvmu(mvmu_id).execute_mvm()

        # return the done event to the caller
        # done_event is a timeout event since this operation takes fixed time
        return self.core.env.timeout(op.accept(self.timing_visitor))

    def visit_hlt(self, op):
        # return the done event to the caller
        # done_event is a timeout event since this operation takes fixed time
        return self.core.env.timeout(op.accept(self.timing_visitor))
