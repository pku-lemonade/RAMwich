from typing import List
from .ima import IMA
from .ops import Load, Set, ALU, MVM, Store, OpType
from .stats import Stats
from .blocks.dram import DRAM
from .blocks.sram import SRAM

logger = logging.getLogger(__name__)

class Core:
    """
    Core in the RAMwich architecture, containing multiple IMAs.
    """
    def __init__(self, id: int, imas: List[IMA], config, dram_capacity: int = 1024):
        self.id = id
        self.imas = imas
        self.config = config
        self.sram = SRAM()
        self.operations: List[OpType] = []
        self.stats = Stats()
        self.dram = DRAM(capacity=dram_capacity)

    def __repr__(self) -> str:
        return f"Core({self.id}, imas={len(self.imas)})"

    def execute_load(self, op: Load) -> bool:
        """Execute a Load operation"""
        try:
            value = self.dram.read(op.d1)
            self.sram.write(0, value)
            self.stats.increment_op_count('load')
            return True
        except IndexError as e:
            print(f"Load operation failed: {e}")
            return False

    def execute_set(self, op: Set) -> bool:
        """Execute a Set operation"""
        try:
            address = self.sram.read(0)
            self.sram.write(1, op.imm)
            self.dram.write(address, op.imm)
            self.stats.increment_op_count('set')
            return True
        except IndexError as e:
            print(f"Set operation failed: {e}")
            return False

    def execute_alu(self, op: ALU) -> bool:
        """Execute an ALU operation"""
        r0 = self.sram.read(0)
        r1 = self.sram.read(1)
        result = 0
        if op.opcode == "add": result = r0 + r1
        elif op.opcode == "sub": result = r0 - r1
        elif op.opcode == "mul": result = r0 * r1
        else:
            print(f"ALU operation failed: Unknown opcode {op.opcode}")
            return False
        self.sram.write(2, result)
        self.stats.increment_op_count('alu')
        return True

    def execute_mvm(self, op: MVM) -> bool:
        """Execute an MVM operation on a specific IMA"""
        if 0 <= op.ima < len(self.imas):
            success = self.imas[op.ima].execute_mvm(op.xbar)
            if success:
                self.stats.increment_op_count('mvm')
            return success
        else:
            print(f"MVM operation failed: Invalid IMA ID {op.ima}")
            return False

    def execute_store(self, op: Store) -> bool:
        """Execute a Store operation"""
        try:
            self.dram.write(op.address, op.value)
            self.stats.increment_op_count('store')
            return True
        except IndexError as e:
            print(f"Store operation failed: {e}")
            return False

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
        for op in self.operations:
            exec_time = 1
            if op.type == 'load':
                exec_time = self.config.load_execution_time
            elif op.type == 'set':
                exec_time = self.config.set_execution_time
            elif op.type == 'alu':
                exec_time = self.config.alu_execution_time
            elif op.type == 'mvm':
                exec_time = self.config.mvm_execution_time

            yield env.timeout(exec_time)

            success = op.accept(self)
            if not success:
                logger.warning(f"Core {self.id}: Operation {op} failed at time {env.now}")
            else:
                logger.debug(f"Core {self.id}: Operation {op} completed at time {env.now}")
                self.stats.latency += exec_time

        logger.info(f"Core {self.id} finished execution at time {env.now}")
