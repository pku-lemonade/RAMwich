from typing import List, Dict, Any, Union
from ima import IMA
from .op import Op
from .stats import Stat
from .dram import DRAM
from .sram import SRAM

class Core:
    """
    Core in the RAMwich architecture, containing multiple IMAs.
    """
    def __init__(self, id: int, imas: List[IMA], dram_capacity: int = 1024):
        self.id = id
        self.imas = imas
        self.sram = SRAM()  # Replace register_file with sram
        self.operations: List[Op] = []
        self.stats = Stat()
        self.dram = DRAM(capacity=dram_capacity)

    def __repr__(self) -> str:
        return f"Core({self.id}, imas={len(self.imas)})"

    def execute_load(self, d1: int) -> bool:
        """Execute a Load operation"""
        # Load data from DRAM address d1 into register 0
        try:
            value = self.dram.read(d1)
            self.sram.write(0, value)
            self.stats.operations += 1
            return True
        except IndexError as e:
            print(f"Load operation failed: {e}")
            return False

    def execute_set(self, imm: int) -> bool:
        """Execute a Set operation"""
        # Set immediate value to register 1
        try:
            address = self.sram.read(0)
            self.sram.write(1, imm)
            self.dram.write(address, imm)
            self.stats.operations += 1
            return True
        except IndexError as e:
            print(f"Set operation failed: {e}")
            return False

    def execute_alu(self, opcode: str) -> bool:
        """Execute an ALU operation"""
        # Placeholder for actual implementation
        r0 = self.sram.read(0)
        r1 = self.sram.read(1)
        result = 0
        if opcode == "add": result = r0 + r1
        elif opcode == "sub": result = r0 - r1
        elif opcode == "mul": result = r0 * r1
        self.sram.write(2, result)
        self.stats.operations += 1
        return True

    def execute_mvm(self, ima_id: int, xbar_ids: List[int]) -> bool:
        """Execute an MVM operation on a specific IMA"""
        if 0 <= ima_id < len(self.imas):
            self.imas[ima_id].execute_mvm(xbar_ids)
            self.stats.operations += 1
            return True
        return False

    def execute_operation(self, op: Op) -> bool:
        """Execute an operation directly on this core"""
        # Let the operation call the appropriate method through its accept method
        result = op.accept(self)

        # Update statistics based on operation type
        self.stats.operations += 1
        return result

    def update_execution_time(self, op_type: str, execution_time: float) -> None:
        """Update the execution time statistics"""
        self.stats.latency += execution_time

        # For MVM operations, also update the IMA stats
        if op_type == 'mvm':
            # Assuming the first IMA is used for simplicity
            ima_id = 0
            self.imas[ima_id].update_execution_time(execution_time)

    def get_stats(self) -> Stat:
        """Get statistics for this Core by aggregating from all components"""
        stats = Stat()

        # Aggregate stats from all components
        components = []
        components.extend(self.imas)  # IMAs will aggregate their own sub-components
        components.append(self.dram)
        components.append(self.sram)

        # Aggregate all component stats
        for component in components:
            component_stats = component.get_stats()
            stats.operations += component_stats.operations
            stats.latency += component_stats.latency
            stats.total_execution_time += component_stats.total_execution_time
            # Add other metrics as needed

        return stats

    def run(self):
        """
        Execute all operations in the operation queue (self.operations)

        Returns:
            Dict containing execution results and updated statistics
        """
        for op in self.operations:
            op.accept(self)
