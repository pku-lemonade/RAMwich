from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class Stats(BaseModel):
    """Generic statistics class for recording latency, energy, and area metrics"""

    dynamic_energy: float = Field(default=0.0, description="Dynamic energy consumption")
    leakage_energy: float = Field(default=0.0, description="Leakage energy consumption")
    area: float = Field(default=0.0, description="Total area")
    op_counts: Dict[str, int] = Field(default_factory=dict, description="Operation counts by type")
    components_activation_count: Dict[str, int] = Field(
        default_factory=dict, description="Components activated times by type"
    )
    components_dynamic_energy_count: Dict[str, float] = Field(
        default_factory=dict, description="Components dynamic energy consumption by type"
    )
    components_leakage_energy_count: Dict[str, float] = Field(
        default_factory=dict, description="Components leakage energy consumption by type"
    )
    components_area_count: Dict[str, float] = Field(default_factory=dict, description="Components area by type")

    def increment_op_count(self, op_type: str, count: int = 1) -> None:
        """Increment the count for a specific operation type"""
        self.op_counts[op_type] = self.op_counts.get(op_type, 0) + count

    def increment_component_activation(self, component_type: str, count: int = 1) -> None:
        """Increment the count for a specific component type"""
        self.components_activation_count[component_type] = (
            self.components_activation_count.get(component_type, 0) + count
        )

    def increment_component_dynamic_energy(self, component_type: str, energy: float) -> None:
        """Increment the energy for a specific component type"""
        self.components_dynamic_energy_count[component_type] = (
            self.components_dynamic_energy_count.get(component_type, 0) + energy
        )
        self.dynamic_energy += energy

    def increment_component_leakage_energy(self, component_type: str, energy: float) -> None:
        """Increment the leakage energy for a specific component type"""
        self.components_leakage_energy_count[component_type] = (
            self.components_leakage_energy_count.get(component_type, 0) + energy
        )
        self.leakage_energy += energy

    def increment_component_area(self, component_type: str, area: float) -> None:
        """Increment the area for a specific component type"""
        self.components_area_count[component_type] = self.components_area_count.get(component_type, 0) + area
        self.area += area

    def calculate_leakage_energy(self, cycles: int) -> None:
        """Calculate the leakage energy based on the number of cycles"""
        self.leakage_energy *= cycles
        for component_type, leakage_energy in self.components_leakage_energy_count.items():
            self.components_leakage_energy_count[component_type] *= cycles

    def print(self):
        """Print the statistics"""
        print("Statistics:")
        print(f"Dynamic Energy: {self.dynamic_energy:.8f} pJ")
        print(f"Leakage Energy: {self.leakage_energy:.8f} pJ")
        print(f"Area: {self.area:.8f} mm^2")
        print("Operation Counts:")
        for op_type, count in self.op_counts.items():
            print(f"  {op_type}: {count}")
        print("Component Activation Counts:")
        for component_type, count in self.components_activation_count.items():
            print(f"  {component_type}: {count}")
        print("Component Dynamic Energy Counts:")
        for component_type, energy in self.components_dynamic_energy_count.items():
            print(f"  {component_type}: {energy:.8f} pJ")
        print("Component Leakage Energy Counts:")
        for component_type, energy in self.components_leakage_energy_count.items():
            print(f"  {component_type}: {energy:.8f} pJ")
        print("Component Area Counts:")
        for component_type, area in self.components_area_count.items():
            print(f"  {component_type}: {area:.8f} mm^2")

    def get_stats(self, components) -> Stats:
        for component in components:
            component_stats = component.get_stats()

            for op_type, count in component_stats.op_counts.items():
                self.increment_op_count(op_type, count)

            for component_type, count in component_stats.components_activation_count.items():
                self.increment_component_activation(component_type, count)

            for component_type, energy in component_stats.components_dynamic_energy_count.items():
                self.increment_component_dynamic_energy(component_type, energy)

            for component_type, energy in component_stats.components_leakage_energy_count.items():
                self.increment_component_leakage_energy(component_type, energy)

            for component_type, area in component_stats.components_area_count.items():
                self.increment_component_area(component_type, area)

        return self
