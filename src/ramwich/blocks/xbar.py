from typing import List

import numpy as np

from pydantic import BaseModel, Field

from ..stats import Stats


class XbarStats(BaseModel):
    operations: int = Field(default=0, description="Total number of operations")
    mvm_operations: int = Field(default=0, description="Number of MVM operations")

    def get_stats(self) -> Stats:
        stats = Stats()
        stats.latency = 0.0
        stats.energy = 0.0
        stats.area = 0.0
        stats.operations = self.operations
        stats.mvm_operations = self.mvm_operations
        return stats


class Xbar:
    """
    Crossbar array component that performs matrix-vector multiplication operations.
    """

    def __init__(self, id: int, size: int = 32):
        self.id = id
        self.size = size
        self.pos_xbar = np.zeros(size, size)
        self.neg_xbar = np.zeros(size, size)
        self.stats = XbarStats()
    
    def load_weights(self, weights: np.ndarray):
        """Load weights into the crossbar"""
        for i in range(self.size):
            for j in range(self.size):
                if weights[i][j] >= 0:
                    self.pos_xbar[i][j] = weights[i][j]
                    self.neg_xbar[i][j] = 0
                else:
                    self.neg_xbar[i][j] = -weights[i][j]
                    self.pos_xbar[i][j] = 0

    def __repr__(self):
        return f"Xbar({self.id}, size={self.size})"

    def execute_mvm(self, xbar_data):
        """Execute a matrix-vector multiplication operation"""
        self.stats.operations += 1
        self.stats.mvm_operations += 1
        return True

    def update_execution_time(self, execution_time):
        """Update the execution time statistics"""
        self.stats.latency += execution_time

    def get_stats(self) -> Stats:
        """Get statistics for this Xbar"""
        return self.stats.get_stats()

    def set_values(self, values: List[int]):
        """Set values in the crossbar"""
        for i, val in enumerate(values):
            if i < self.size:
                self.memory[i] = val
            else:
                break
