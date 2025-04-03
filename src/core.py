from typing import List, Dict, Any, Union
from ima import IMA
from .op import Op
from .stats import Stat  # CoreStats is no longer needed

class Core:
    """
    Core in the RAMwich architecture, containing multiple IMAs.
    """
    def __init__(self, id: int, imas: List[IMA]):
        self.id = id
        self.imas = imas
        self.registers: List[int] = [0] * 16  # Default 16 registers
        self.operations: List[Op] = []  # Store operations to be executed
        self.stats = Stat()

    def __repr__(self) -> str:
        return f"Core({self.id}, imas={len(self.imas)})"

    def execute_load(self, d1: int) -> bool:
        """Execute a Load operation"""
        # Placeholder for actual implementation
        self.registers[0] = d1
        self.stats.operations += 1
        return True

    def execute_set(self, imm: int) -> bool:
        """Execute a Set operation"""
        # Placeholder for actual implementation
        self.registers[1] = imm
        self.stats.operations += 1
        return True

    def execute_alu(self, opcode: str) -> bool:
        """Execute an ALU operation"""
        # Placeholder for actual implementation
        if opcode == "add":
            self.registers[2] = self.registers[0] + self.registers[1]
        elif opcode == "sub":
            self.registers[2] = self.registers[0] - self.registers[1]
        elif opcode == "mul":
            self.registers[2] = self.registers[0] * self.registers[1]
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

    def get_stats(self, include_components: bool = True) -> Stat:
        """Get statistics for this Core and optionally its components"""
        # Directly modify and return the original stats object
        return self.stats.get_stats(
            components=self.imas if include_components else None
        )

    def run(self):
        """
        Execute all operations in the operation queue (self.operations)

        Returns:
            Dict containing execution results and updated statistics
        """
        for op in self.operations:
            op.accept(self)
