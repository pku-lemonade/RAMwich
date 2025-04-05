from typing import List
from .ima import IMA
from .ops import Load, Set, ALU, MVM, Store, CoreOp
from .stats import Stats
from .blocks.dram import DRAM
from .blocks.sram import SRAM

logger = logging.getLogger(__name__)

class CoreTiming:
    def __init__(self, config):
        self.config = config

    def visit_load(self, op):
        return self.config.load_execution_time

    def visit_set(self, op):
        return self.config.set_execution_time

    def visit_alu(self, op):
        return self.config.alu_execution_time

    def visit_mvm(self, op):
        return self.config.mvm_execution_time

    def visit_store(self, op):
        return self.config.load_execution_time  # Assuming store uses load time

class CoreExecution:
    def __init__(self, core):
        self.core = core

    def visit_load(self, op):
        try:
            value = self.core.dram.read(op.d1)
            self.core.sram.write(0, value)
            self.core.stats.increment_op_count('load')
            return True
        except IndexError as e:
            print(f"Load operation failed: {e}")
            return False

    def visit_set(self, op):
        try:
            address = self.core.sram.read(0)
            self.core.sram.write(1, op.imm)
            self.core.dram.write(address, op.imm)
            self.core.stats.increment_op_count('set')
            return True
        except IndexError as e:
            print(f"Set operation failed: {e}")
            return False

    def visit_alu(self, op):
        r0 = self.core.sram.read(0)
        r1 = self.core.sram.read(1)
        result = 0
        if op.opcode == "add": result = r0 + r1
        elif op.opcode == "sub": result = r0 - r1
        elif op.opcode == "mul": result = r0 * r1
        else:
            print(f"ALU operation failed: Unknown opcode {op.opcode}")
            return False
        self.core.sram.write(2, result)
        self.core.stats.increment_op_count('alu')
        return True

    def visit_mvm(self, op):
        if 0 <= op.ima < len(self.core.imas):
            success = self.core.imas[op.ima].execute_mvm(op.xbar)
            if success:
                self.core.stats.increment_op_count('mvm')
            return success
        else:
            print(f"MVM operation failed: Invalid IMA ID {op.ima}")
            return False

    def visit_store(self, op):
        try:
            self.core.dram.write(op.address, op.value)
            self.core.stats.increment_op_count('store')
            return True
        except IndexError as e:
            print(f"Store operation failed: {e}")
            return False

class Core:
    """
    Core in the RAMwich architecture, containing multiple IMAs.
    """
    def __init__(self, id: int, imas: List[IMA], config, dram_capacity: int = 1024):
        self.id = id
        self.imas = imas
        self.config = config
        self.sram = SRAM()
        self.operations: List[CoreOp] = []
        self.stats = Stats()
        self.dram = DRAM(capacity=dram_capacity)

    def __repr__(self) -> str:
        return f"Core({self.id}, imas={len(self.imas)})"

    def get_stats(self) -> Stats:
        """Get statistics for this Core by aggregating from all components"""
        aggregated_stats = self.stats.copy(deep=True)
        components = []
        components.extend(self.imas)
        components.append(self.dram)
        components.append(self.sram)

        for component in components:
            if hasattr(component, 'get_stats'):
                component_stats = component.get_stats()
                aggregated_stats.latency += component_stats.latency
                aggregated_stats.energy += component_stats.energy
                aggregated_stats.area += component_stats.area
                for op_type, count in component_stats.op_counts.items():
                    aggregated_stats.increment_op_count(op_type, count)

        return aggregated_stats

    def run(self, env):
        """
        Execute all operations assigned to this core.
        This method should be called as a SimPy process.
        """
        logger.info(f"Core {self.id} starting execution at time {env.now}")

        timing_visitor = CoreTiming(self.config)
        execution_visitor = CoreExecution(self)

        for op in self.operations:
            exec_time = op.accept(timing_visitor)
            yield env.timeout(exec_time)

            success = op.accept(execution_visitor)
            if not success:
                logger.warning(f"Core {self.id}: Operation {op} failed at time {env.now}")
            else:
                logger.debug(f"Core {self.id}: Operation {op} completed at time {env.now}")
                self.stats.latency += exec_time

        logger.info(f"Core {self.id} finished execution at time {env.now}")
