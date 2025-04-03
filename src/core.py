from typing import List
from ima import IMA

class Core:
    """
    Core in the RAMwich architecture, containing multiple IMAs.
    """
    def __init__(self, id: int, imas: List[IMA]):
        self.id = id
        self.imas = imas
        self.registers = [0] * 16  # Default 16 registers
        self.operations = []  # Store operations to be executed
        self.stats = {
            'operations': 0,
            'load_operations': 0,
            'set_operations': 0,
            'alu_operations': 0,
            'mvm_operations': 0,
            'total_execution_time': 0,
            'last_execution_time': 0
        }

    def __repr__(self):
        return f"Core({self.id}, imas={len(self.imas)})"

    def execute_load(self, d1: int):
        """Execute a Load operation"""
        # Placeholder for actual implementation
        self.registers[0] = d1
        self.stats['operations'] += 1
        self.stats['load_operations'] += 1
        return True

    def execute_set(self, imm: int):
        """Execute a Set operation"""
        # Placeholder for actual implementation
        self.registers[1] = imm
        self.stats['operations'] += 1
        self.stats['set_operations'] += 1
        return True

    def execute_alu(self, opcode: str):
        """Execute an ALU operation"""
        # Placeholder for actual implementation
        if opcode == "add":
            self.registers[2] = self.registers[0] + self.registers[1]
        elif opcode == "sub":
            self.registers[2] = self.registers[0] - self.registers[1]
        elif opcode == "mul":
            self.registers[2] = self.registers[0] * self.registers[1]
        self.stats['operations'] += 1
        self.stats['alu_operations'] += 1
        return True

    def execute_mvm(self, ima_id, xbar_ids):
        """Execute an MVM operation on a specific IMA"""
        if 0 <= ima_id < len(self.imas):
            self.imas[ima_id].execute_mvm(xbar_ids)
            self.stats['operations'] += 1
            self.stats['mvm_operations'] += 1
            return True
        return False

    def update_execution_time(self, op_type, execution_time):
        """Update the execution time statistics"""
        self.stats['total_execution_time'] += execution_time
        self.stats['last_execution_time'] = execution_time

        # For MVM operations, also update the IMA stats
        if op_type == 'mvm':
            # Assuming the first IMA is used for simplicity
            ima_id = 0
            self.imas[ima_id].update_execution_time(execution_time)

    def get_stats(self, include_components=True):
        """Get statistics for this Core and optionally its components"""
        result = {
            'core_id': self.id,
            'stats': self.stats.copy()
        }

        if include_components:
            result['imas'] = [
                ima.get_stats(include_components)
                for ima in self.imas
            ]

        return result
