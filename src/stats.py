from typing import Dict, Any, List, Optional, TypeVar, Protocol, runtime_checkable, Union
from pydantic import BaseModel, Field

class Stats(BaseModel):
    """Generic statistics class for recording latency, energy, and area metrics"""

    # Basic metrics
    latency: float = Field(default=0.0, description="Total latency/execution time")
    energy: float = Field(default=0.0, description="Total energy consumption")
    area: float = Field(default=0.0, description="Total area")

    # Operation counter
    op_counts: Dict[str, int] = Field(default_factory=dict, description="Operation counts by type")

    def increment_op_count(self, op_type: str, count: int = 1) -> None:
        """Increment the count for a specific operation type"""
        self.op_counts[op_type] = self.op_counts.get(op_type, 0) + count

    def get_stats(self, components):
        # Get stats from each subcomponent
        for component in components:
            if hasattr(component, 'get_stats'):
                # Recursively gather stats from subcomponents
                component_stats = component.get_stats()

                # Aggregate the stats
                self.latency += component_stats.latency
                self.energy += component_stats.energy
                self.area += component_stats.area

                # Aggregate operation counts from dictionary
                for op_type, count in component_stats.op_counts.items():
                    self.increment_op_count(op_type, count)

        return self
