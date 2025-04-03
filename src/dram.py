from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .stats import Stat

class DramStats(BaseModel):
    """Statistics for DRAM operations"""
    read_operations: int = Field(default=0, description="Number of read operations")
    write_operations: int = Field(default=0, description="Number of write operations")
    total_operations: int = Field(default=0, description="Total number of operations")

    def get_stats(self) -> Stat:
        """Convert DramStats to general Stat object"""
        stats = Stat()
        stats.latency = 0.0  # Will be updated through update_execution_time
        stats.energy = 0.0  # Placeholder for energy consumption
        stats.area = 0.0    # Placeholder for area usage
        stats.operations = self.total_operations
        stats.read_operations = self.read_operations
        stats.write_operations = self.write_operations
        return stats

class DRAM:
    """
    DRAM module for memory operations in the RAMwich architecture.
    """
    def __init__(self, capacity: int = 1024, latency: float = 0.1):
        """
        Initialize DRAM with specified capacity.

        Args:
            capacity: Memory capacity in units (default: 1024)
            latency: Access latency in time units (default: 0.1)
        """
        self.capacity = capacity
        self.memory = [0] * capacity
        self.latency = latency
        self.stats = DramStats()

    def __repr__(self) -> str:
        return f"DRAM(capacity={self.capacity}, used={sum(1 for v in self.memory if v != 0)})"

    def read(self, address: int) -> int:
        """
        Read value from specified memory address.

        Args:
            address: Memory address to read from

        Returns:
            Value at the memory address
        """
        if 0 <= address < self.capacity:
            self._update_stats(operation_type="read")
            return self.memory[address]
        raise IndexError(f"Memory address {address} out of range (0-{self.capacity-1})")

    def write(self, address: int, value: int) -> bool:
        """
        Write value to specified memory address.

        Args:
            address: Memory address to write to
            value: Value to write

        Returns:
            True if write successful, False otherwise
        """
        if 0 <= address < self.capacity:
            self.memory[address] = value
            self._update_stats(operation_type="write")
            return True
        raise IndexError(f"Memory address {address} out of range (0-{self.capacity-1})")

    def _update_stats(self, operation_type: str) -> None:
        """Update statistics based on operation type"""
        self.stats.total_operations += 1

        if operation_type == "read":
            self.stats.read_operations += 1
        elif operation_type == "write":
            self.stats.write_operations += 1

    def update_execution_time(self, execution_time: float) -> None:
        """Update the execution time statistics"""
        self.stats.latency += execution_time

    def get_stats(self) -> Stat:
        """Get statistics for this DRAM module"""
        return self.stats.get_stats()
