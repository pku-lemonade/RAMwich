from __future__ import annotations

from typing import Dict, List, TypedDict

from pydantic import BaseModel, Field


class Stats(BaseModel):
    activation_count: int = Field(default=0, description="Activation count")
    dynamic_energy: float = Field(default=0.0, description="Dynamic energy consumption")
    leakage_energy: float = Field(default=0.0, description="Leakage energy consumption")
    area: float = Field(default=0.0, description="Total area")

    def merge(self, other: Stats) -> Stats:
        """Merge another Stats object into this one"""
        self.activation_count += other.activation_count
        self.dynamic_energy += other.dynamic_energy
        self.leakage_energy += other.leakage_energy
        self.area += other.area
        return self


class StatsDict(dict):
    """Typed dictionary for storing statistics"""

    def merge(self, other: StatsDict) -> StatsDict:
        """Merge another StatsDict object into this one"""
        for key, value in other.items():
            if key in self:
                self[key].merge(value)
            else:
                self[key] = value
        return self

    def update_leakage_energy(self, cycles: int) -> StatsDict:
        """Update the leakage energy for all components"""
        for key, value in self.items():
            value.leakage_energy *= cycles
        return self

    def summary(self) -> Stats:
        """Generate a summary of the statistics"""
        result = Stats()
        for key, value in self.items():
            result.merge(value)
        return result

    def print(self):
        """Print the statistics in a readable format"""
        print("Statistics Summary:")
        print(f"Total Dynamic Energy: {self.summary().dynamic_energy} J")
        print(f"Total Leakage Energy: {self.summary().leakage_energy} J")
        print(f"Total Area: {self.summary().area} mm^2")
        for key, value in self.items():
            print(f"{key}: {value}")
