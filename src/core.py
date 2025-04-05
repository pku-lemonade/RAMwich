from typing import List, Dict, Any, Union, TYPE_CHECKING
from .ima import IMA
# Import specific operation types
from .ops import Load, Set, ALU, MVM, Store, OpType
from .stats import Stats
from .blocks.dram import DRAM
from .blocks.sram import SRAM

class Core:
    """
    Core in the RAMwich architecture, containing multiple IMAs.
    """
    def __init__(self, id: int, imas: List[IMA], dram_capacity: int = 1024):
        self.id = id
        self.imas = imas
        self.sram = SRAM()  # Replace register_file with sram
        self.operations: List[OpType] = [] # Use the union type OpType
        self.stats = Stats()
        self.dram = DRAM(capacity=dram_capacity)

    def __repr__(self) -> str:
        return f"Core({self.id}, imas={len(self.imas)})"

    def execute_load(self, op: Load) -> bool:
        """Execute a Load operation"""
        # Load data from DRAM address op.d1 into register 0
        try:
            value = self.dram.read(op.d1)
            self.sram.write(0, value)
            # TODO: Update stats based on config execution time
            # self.update_execution_time('load', execution_time)
            self.stats.increment_op_count('load')
            return True
        except IndexError as e:
            print(f"Load operation failed: {e}")
            return False

    def execute_set(self, op: Set) -> bool:
        """Execute a Set operation"""
        # Set immediate value to register 1, write to DRAM address from reg 0
        try:
            address = self.sram.read(0)
            self.sram.write(1, op.imm)
            self.dram.write(address, op.imm)
            # TODO: Update stats based on config execution time
            # self.update_execution_time('set', execution_time)
            self.stats.increment_op_count('set')
            return True
        except IndexError as e:
            print(f"Set operation failed: {e}")
            return False

    def execute_alu(self, op: ALU) -> bool:
        """Execute an ALU operation"""
        # Placeholder for actual implementation
        r0 = self.sram.read(0)
        r1 = self.sram.read(1)
        result = 0
        if op.opcode == "add": result = r0 + r1
        elif op.opcode == "sub": result = r0 - r1
        elif op.opcode == "mul": result = r0 * r1
        # Add other ALU opcodes as needed
        else:
            print(f"ALU operation failed: Unknown opcode {op.opcode}")
            return False
        self.sram.write(2, result)
        # TODO: Update stats based on config execution time
        # self.update_execution_time('alu', execution_time)
        self.stats.increment_op_count('alu')
        return True

    def execute_mvm(self, op: MVM) -> bool:
        """Execute an MVM operation on a specific IMA"""
        if 0 <= op.ima < len(self.imas):
            # Pass xbar data (op.xbar) to the IMA's execute_mvm
            # Assuming IMA's execute_mvm expects the xbar data list
            success = self.imas[op.ima].execute_mvm(op.xbar)
            if success:
                # TODO: Update stats based on config execution time
                # self.update_execution_time('mvm', execution_time)
                self.stats.increment_op_count('mvm')
                # IMA stats are updated within its own execute_mvm or similar method
            return success
        else:
            print(f"MVM operation failed: Invalid IMA ID {op.ima}")
            return False

    def execute_store(self, op: Store) -> bool:
        """Execute a Store operation"""
        # Store value op.value to DRAM address op.address
        try:
            self.dram.write(op.address, op.value)
            # TODO: Update stats based on config execution time
            # self.update_execution_time('store', execution_time)
            self.stats.increment_op_count('store')
            return True
        except IndexError as e:
            print(f"Store operation failed: {e}")
            return False

    def get_stats(self) -> Stats:
        """Get statistics for this Core by aggregating from all components"""
        # Start with the core's own stats (op_counts)
        aggregated_stats = self.stats.copy(deep=True)

        # Aggregate stats from all components
        components = []
        components.extend(self.imas)
        components.append(self.dram)
        components.append(self.sram)

        # Aggregate all component stats
        for component in components:
            if hasattr(component, 'get_stats'):
                component_stats = component.get_stats()
                aggregated_stats.latency += component_stats.latency
                aggregated_stats.energy += component_stats.energy
                aggregated_stats.area += component_stats.area
                # Aggregate op_counts carefully
                for op_type, count in component_stats.op_counts.items():
                    aggregated_stats.increment_op_count(op_type, count)

        return aggregated_stats

    def run(self, simulator, env):
        """
        Execute all operations assigned to this core.
        This method should be called as a SimPy process.
        """
        logger.info(f"Core {self.id} starting execution at time {env.now}")
        for op in self.operations:
            # Get execution time from config (assuming simulator has access)
            # This part needs refinement based on how config times are accessed
            exec_time = 1 # Placeholder
            if op.type == 'load':
                exec_time = simulator.config.load_execution_time
            elif op.type == 'set':
                exec_time = simulator.config.set_execution_time
            elif op.type == 'alu':
                exec_time = simulator.config.alu_execution_time
            elif op.type == 'mvm':
                exec_time = simulator.config.mvm_execution_time
            # Add other op types like store if needed

            # Yield timeout for the operation execution time
            yield env.timeout(exec_time)

            # Execute the operation using the visitor pattern
            success = op.accept(self)
            if not success:
                logger.warning(f"Core {self.id}: Operation {op} failed at time {env.now}")
            else:
                logger.debug(f"Core {self.id}: Operation {op} completed at time {env.now}")
                # Update latency in core stats
                self.stats.latency += exec_time

        logger.info(f"Core {self.id} finished execution at time {env.now}")
