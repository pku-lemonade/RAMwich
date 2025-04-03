from typing import List
from pydantic import BaseModel, Field
from .stats import Stat

class SRAMStats(BaseModel):
    """Statistics for SRAM operations"""
    read_operations: int = Field(default=0, description="Number of read operations")
    write_operations: int = Field(default=0, description="Number of write operations")
    total_operations: int = Field(default=0, description="Total number of operations")
    total_execution_time: float = Field(default=0, description="Total execution time")

    def get_stats(self) -> Stat:
        """Convert SRAMStats to general Stat object"""
        stats = Stat()
        stats.latency = float(self.total_execution_time)
        stats.operations = self.total_operations
        stats.read_operations = self.read_operations
        stats.write_operations = self.write_operations
        stats.total_execution_time = float(self.total_execution_time)
        return stats

class SRAM:
    """SRAM register file component for the Core"""
    def __init__(self, size: int = 16, latency: float = 0.01):
        self.size = size
        self.registers = [0] * size
        self.latency = latency
        self.stats = SRAMStats()

    def read(self, reg_id: int) -> int:
        if 0 <= reg_id < self.size:
            self._update_stats("read")
            return self.registers[reg_id]
        raise IndexError(f"Register {reg_id} out of range (0-{self.size-1})")

    def write(self, reg_id: int, value: int) -> None:
        if 0 <= reg_id < self.size:
            self.registers[reg_id] = value
            self._update_stats("write")
        else:
            raise IndexError(f"Register {reg_id} out of range (0-{self.size-1})")

    def _update_stats(self, operation_type: str) -> None:
        self.stats.total_operations += 1
        if operation_type == "read":
            self.stats.read_operations += 1
        elif operation_type == "write":
            self.stats.write_operations += 1
        self.stats.total_execution_time += self.latency

    def get_stats(self) -> Stat:
        return self.stats.get_stats()
