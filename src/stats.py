from typing import Dict, Any, List, Optional
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

    def record_operation(self, op_latency: float = 0.0, op_energy: float = 0.0, op_type: Optional[str] = None):
        """Record a single operation's metrics"""
        self.operations += 1
        self.latency += op_latency
        self.energy += op_energy

        # Update operation-specific counters if type is provided
        if op_type:
            if op_type == 'load':
                self.load_operations += 1
            elif op_type == 'set':
                self.set_operations += 1
            elif op_type == 'alu':
                self.alu_operations += 1
            elif op_type == 'mvm':
                self.mvm_operations += 1

    def get_stats(self,
                 component_id: Optional[int] = None,
                 component_type: Optional[str] = None,
                 components: Optional[List['Stat']] = None,
                 include_components: bool = True) -> 'Stat':
        """
        Aggregate statistics from subcomponents and update self

        Args:
            component_id: ID of the component (not used for aggregation)
            component_type: Type of component (not used for aggregation)
            components: List of subcomponent Stat objects to include and aggregate
            include_components: Whether to include subcomponent stats

        Returns:
            Self, with updated aggregated statistics
        """
        # If components are provided, aggregate their stats into self
        if components and include_components:
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

    # You might want a method that returns a dictionary representation
    def to_dict(self,
                component_id: Optional[int] = None,
                component_type: Optional[str] = None) -> Dict[str, Any]:
        """Convert stats to dictionary representation"""
        result = {
            'stats': {
                'latency': self.latency,
                'energy': self.energy,
                'area': self.area,
                'operations': self.operations,
                'load_operations': self.load_operations,
                'set_operations': self.set_operations,
                'alu_operations': self.alu_operations,
                'mvm_operations': self.mvm_operations,
                'total_execution_time': self.total_execution_time,
                'last_execution_time': self.last_execution_time
            }
        }

        # Add component identifier if provided
        if component_id is not None:
            if component_type:
                result[f'{component_type}_id'] = component_id
            else:
                result['id'] = component_id

        return result
