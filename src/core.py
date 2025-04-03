from typing import List, Dict, Any, Union
from ima import IMA
from .op import Op
from pydantic import BaseModel, Field

class CoreStats(BaseModel):
    operations: int = Field(default=0, description="Total number of operations")
    load_operations: int = Field(default=0, description="Number of load operations")
    set_operations: int = Field(default=0, description="Number of set operations")
    alu_operations: int = Field(default=0, description="Number of ALU operations")
    mvm_operations: int = Field(default=0, description="Number of MVM operations")
    total_execution_time: float = Field(default=0, description="Total execution time")
    last_execution_time: float = Field(default=0, description="Last operation execution time")

    def get_stats(self, core_id: int, include_components: bool = True, imas=None) -> Dict[str, Any]:
        """Get statistics for this Core and optionally its components"""
        result = {
            'core_id': core_id,
            'stats': self.dict()
        }

        if include_components and imas:
            result['imas'] = [
                ima.get_stats(include_components)
                for ima in imas
            ]

        return result

class Core:
    """
    Core in the RAMwich architecture, containing multiple IMAs.
    """
    def __init__(self, id: int, imas: List[IMA]):
        self.id = id
        self.imas = imas
        self.registers: List[int] = [0] * 16  # Default 16 registers
        self.operations: List[Op] = []  # Store operations to be executed
        self.stats = CoreStats()

    def __repr__(self) -> str:
        return f"Core({self.id}, imas={len(self.imas)})"

    def execute_load(self, d1: int) -> bool:
        """Execute a Load operation"""
        # Placeholder for actual implementation
        self.registers[0] = d1
        self.stats.operations += 1
        self.stats.load_operations += 1
        return True

    def execute_set(self, imm: int) -> bool:
        """Execute a Set operation"""
        # Placeholder for actual implementation
        self.registers[1] = imm
        self.stats.operations += 1
        self.stats.set_operations += 1
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
        self.stats.alu_operations += 1
        return True

    def execute_mvm(self, ima_id: int, xbar_ids: List[int]) -> bool:
        """Execute an MVM operation on a specific IMA"""
        if 0 <= ima_id < len(self.imas):
            self.imas[ima_id].execute_mvm(xbar_ids)
            self.stats.operations += 1
            self.stats.mvm_operations += 1
            return True
        return False

    def execute_operation(self, op: Op) -> bool:
        """Execute an operation directly on this core"""
        # Let the operation call the appropriate method through its accept method
        result = op.accept(self)

        # Update statistics based on operation type
        op_type = op.__class__.__name__.lower()
        self.stats.operations += 1
        if op_type in ['load', 'set', 'alu', 'mvm']:
            self.stats[f'{op_type}_operations'] += 1

        return result

    def update_execution_time(self, op_type: str, execution_time: float) -> None:
        """Update the execution time statistics"""
        self.stats.total_execution_time += execution_time
        self.stats.last_execution_time = execution_time

        # For MVM operations, also update the IMA stats
        if op_type == 'mvm':
            # Assuming the first IMA is used for simplicity
            ima_id = 0
            self.imas[ima_id].update_execution_time(execution_time)

    def get_stats(self, include_components: bool = True) -> Dict[str, Any]:
        """Get statistics for this Core and optionally its components"""
        return self.stats.get_stats(self.id, include_components, self.imas)

    def run(self) -> Dict[str, Any]:
        """
        Execute all operations in the operation queue (self.operations)

        Returns:
            Dict containing execution results and updated statistics
        """
        results = []

        for op in self.operations:
            result = self.execute_operation(op)
            results.append(result)

        return {
            'success': all(results),
            'operations_executed': len(results),
            'stats': self.get_stats(include_components=False)
        }
