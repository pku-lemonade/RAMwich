from typing import Dict, Any, List, Optional, TypeVar, Protocol, runtime_checkable, Union
from pydantic import BaseModel, Field

class Stat(BaseModel):
    """Generic statistics class for recording latency, energy, and area metrics"""

    # Basic metrics
    latency: float = Field(default=0.0, description="Total latency/execution time")
    energy: float = Field(default=0.0, description="Total energy consumption")
    area: float = Field(default=0.0, description="Total area")

    # Operation counter
    operations: int = Field(default=0, description="Total number of operations")

    # Core-specific operation counters
    load_operations: int = Field(default=0, description="Number of load operations")
    set_operations: int = Field(default=0, description="Number of set operations")
    alu_operations: int = Field(default=0, description="Number of ALU operations")
    mvm_operations: int = Field(default=0, description="Number of MVM operations")

    # Execution time metrics
    total_execution_time: float = Field(default=0.0, description="Total execution time")
    last_execution_time: float = Field(default=0.0, description="Last operation execution time")

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
                self.operations += component_stats.operations
                self.load_operations += component_stats.load_operations
                self.set_operations += component_stats.set_operations
                self.alu_operations += component_stats.alu_operations
                self.mvm_operations += component_stats.mvm_operations
                self.total_execution_time += component_stats.total_execution_time
                # We don't aggregate last_execution_time as it's not cumulative

        return self
