from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import Config
from ..stats import Stats


class VFUStats(BaseModel):
    """Statistics for VFU operations"""

    mul_operations: int = Field(default=0, description="Number of read operations")
    div_operations: int = Field(default=0, description="Number of write operations")
    act_operations: int = Field(default=0, description="Number of activation operations")
    other_operations: int = Field(default=0, description="Number of other operations")
    total_operations: int = Field(default=0, description="Total number of operations")

    def get_stats(self) -> Stats:
        """Convert SRAMStats to general Stats object"""
        stats = Stats()
        stats.latency = 0.0  # Will be updated through execution
        return stats


class VFU:
    """SRAM register file component for the Core"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.frac_bits = self.config.data_config.frac_bits

        # Initialize stats
        self.stats = VFUStats()

    def calculate(self, a: Union[NDArray[np.integer], int], b: Optional[Union[NDArray[np.integer], int]]):
        """Perform a calculation using the ALU and activation unit"""
        # Example operation: multiplication
        if isinstance(a, np.ndarray):
            result = np.multiply(a, b)
        else:
            result = a * b

        # Update stats
        self._update_stats("mul")

        return result

    def _update_stats(self, operation_type: str) -> None:
        self.stats.total_operations += 1
        if operation_type == "read":
            self.stats.read_operations += 1
        elif operation_type == "write":
            self.stats.write_operations += 1
        self.stats.latency += self.latency

    def get_stats(self) -> Stats:
        return self.stats.get_stats()
