from typing import Dict, Any, List
from pydantic import BaseModel, Field
from .stats import Stat

class XbarStats(BaseModel):
    operations: int = Field(default=0, description="Total number of operations")
    mvm_operations: int = Field(default=0, description="Number of MVM operations")
    total_execution_time: float = Field(default=0, description="Total execution time")
    last_execution_time: float = Field(default=0, description="Last operation execution time")

    def get_stats(self, xbar_id: int) -> Stat:
        """Get statistics for this Xbar"""
        stats = Stat()

        # Map Xbar metrics to Stat object
        stats.latency = float(self.total_execution_time)
        stats.energy = 0.0  # Set appropriate energy value if available
        stats.area = 0.0    # Set appropriate area value if available

        # Map operation counts
        stats.operations = self.operations
        stats.mvm_operations = self.mvm_operations

        # Set execution time metrics
        stats.total_execution_time = float(self.total_execution_time)
        stats.last_execution_time = float(self.last_execution_time)

        return stats

class Xbar:
    """
    Crossbar array component that performs matrix-vector multiplication operations.
    """
    def __init__(self, id: int, size: int = 32):
        self.id = id
        self.size = size
        self.memory = [0] * size
        self.stats = XbarStats()

    def __repr__(self):
        return f"Xbar({self.id}, size={self.size})"

    def execute_mvm(self, xbar_data):
        """Execute a matrix-vector multiplication operation"""
        self.stats.operations += 1
        self.stats.mvm_operations += 1
        # Actual implementation would be more complex
        return True

    def update_execution_time(self, execution_time):
        """Update the execution time statistics"""
        self.stats.total_execution_time += execution_time
        self.stats.last_execution_time = execution_time

    def get_stats(self) -> Stat:
        """Get statistics for this Xbar"""
        return self.stats.get_stats(self.id)

    def set_values(self, values: List[int]):
        """Set values in the crossbar"""
        for i, val in enumerate(values):
            if i < self.size:
                self.memory[i] = val
            else:
                break
