from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Stats(BaseModel):
    """Generic statistics class for recording latency, energy, and area metrics"""

    latency: float = Field(default=0.0, description="Total latency/execution time")
    energy: float = Field(default=0.0, description="Total energy consumption")
    area: float = Field(default=0.0, description="Total area")
    op_counts: Dict[str, int] = Field(default_factory=dict, description="Operation counts by type")
    completion_times: List[float] = Field(default_factory=list, description="Timestamp of each operation completion")
    total_execution_time: float = Field(default=0.0, description="Total execution time for the component")

    def increment_op_count(self, op_type: str, count: int = 1) -> None:
        """Increment the count for a specific operation type"""
        self.op_counts[op_type] = self.op_counts.get(op_type, 0) + count

    def get_stats(self, components) -> Stats:
        for component in components:
            component_stats = component.get_stats()
            self.latency += component_stats.latency
            self.energy += component_stats.energy
            self.area += component_stats.area
            for op_type, count in component_stats.op_counts.items():
                self.increment_op_count(op_type, count)
        return self
