from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class Stats(BaseModel):
    """Generic statistics class for recording latency, energy, and area metrics"""

    dynamic_energy: float = Field(default=0.0, description="Dynamic energy consumption")
    leakage_energy: float = Field(default=0.0, description="Leakage energy consumption")
    area: float = Field(default=0.0, description="Total area")
    op_counts: Dict[str, int] = Field(default_factory=dict, description="Operation counts by type")
    components_activated_count: int = Field(default_factory=dict, description="Components activated times by type")

    def increment_op_count(self, op_type: str, count: int = 1) -> None:
        """Increment the count for a specific operation type"""
        self.op_counts[op_type] = self.op_counts.get(op_type, 0) + count

    def increment_component_count(self, component_type: str, count: int = 1) -> None:
        """Increment the count for a specific component type"""
        self.components_activated_count[component_type] = self.components_activated_count.get(component_type, 0) + count

    def get_stats(self, components) -> Stats:
        for component in components:
            component_stats = component.get_stats()
            self.dynamic_energy += component_stats.dynamic_energy
            self.leakage_energy += component_stats.leakage_energy
            self.area += component_stats.area
            for op_type, count in component_stats.op_counts.items():
                self.increment_op_count(op_type, count)
            for component_type, count in component_stats.components_activated_count.items():
                self.increment_component_count(component_type, count)
        return self
