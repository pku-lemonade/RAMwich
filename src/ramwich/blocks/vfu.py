from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import Config
from ..ops import VFUOpType
from ..stats import Stats


class VFUStats(BaseModel):
    """Statistics for VFU operations"""

    # Universal metrics
    unit_energy_consumption_mul: float = Field(default=0.0, description="Energy consumption for each convertion in pJ")
    unit_energy_consumption_div: float = Field(default=0.0, description="Energy consumption for each convertion in pJ")
    unit_energy_consumption_act: float = Field(default=0.0, description="Energy consumption for each convertion in pJ")
    unit_energy_consumption_others: float = Field(
        default=0.0, description="Energy consumption for each convertion in pJ"
    )
    leakage_energy_per_cycle: float = Field(default=0.0, description="Leakage energy consumption for 1 cycle in pJ")
    area = float = Field(default=0.0, description="Area in mm^2")

    # VFU specific metrics
    mul_operations: int = Field(default=0, description="Number of read operations")
    div_operations: int = Field(default=0, description="Number of write operations")
    act_operations: int = Field(default=0, description="Number of activation operations")
    other_operations: int = Field(default=0, description="Number of other operations")
    total_operations: int = Field(default=0, description="Total number of operations")

    def get_stats(self) -> Stats:
        """Convert SRAMStats to general Stats object"""
        stats = Stats()

        stats.dynamic_energy = (
            self.unit_energy_consumption_mul * self.mul_operations
            + self.unit_energy_consumption_div * self.div_operations
            + self.unit_energy_consumption_act * self.act_operations
            + self.unit_energy_consumption_others * self.other_operations
        )
        stats.leakage_energy = self.leakage_energy_per_cycle
        stats.area = self.area

        stats.increment_component_count("VFU", self.total_operations)
        stats.increment_component_count("VFU MUL", self.mul_operations)
        stats.increment_component_count("VFU DIV", self.div_operations)
        stats.increment_component_count("VFU ACT", self.act_operations)
        stats.increment_component_count("VFU OTHERS", self.other_operations)

        return stats


class VFU:
    """SRAM register file component for the Core"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.frac_bits = self.config.data_config.frac_bits
        self.min_value = -1 * (1 << self.config.data_config.data_bits)
        self.max_value = (1 << self.config.data_config.data_bits) - 1

        # Initialize operation handlers
        self._init_op_handlers()

        # Initialize stats
        self.stats = VFUStats()
        self.stats.unit_energy_consumption_mul = self.config.core_config.alu_pow_mul_dyn
        self.stats.unit_energy_consumption_div = self.config.core_config.alu_pow_div_dyn
        self.stats.unit_energy_consumption_act = self.config.core_config.act_pow_dyn
        self.stats.unit_energy_consumption_others = self.config.core_config.alu_pow_others_dyn
        self.stats.leakage_energy_per_cycle = self.config.core_config.alu_pow_leak
        self.stats.area = self.config.core_config.vfu_area

    def _init_op_handlers(self):
        """Initialize operation handlers dictionary"""
        self.op_handlers = {
            "and": self._handle_and,
            "or": self._handle_or,
            "not": self._handle_not,
            "add": self._handle_add,
            "sub": self._handle_sub,
            "mul": self._handle_mul,
            "div": self._handle_div,
            "min": self._handle_min,
            "max": self._handle_max,
            "sig": self._handle_sig,
            "tanh": self._handle_tanh,
            "relu": self._handle_relu,
        }

    def calculate(
        self, opcode: VFUOpType, a: Union[NDArray[np.int32], int], b: Optional[Union[NDArray[np.int32], int]] = None
    ) -> NDArray[np.int32]:
        """Perform a calculation using the ALU and activation unit"""
        # Example operation: multiplication
        if isinstance(a, int):
            a = np.array([a], dtype=np.int32)

        length = len(a)

        if b is not None and isinstance(b, int):
            b = np.array([b], dtype=np.int32)
            if len(b) != length:
                raise ValueError("Operands must be of the same length")

        # Determine operation type for statistics
        if opcode in ["mul", "div"]:
            operation_type = opcode
        elif opcode in ["relu", "sig", "tanh"]:
            operation_type = "act"
        else:
            operation_type = "other"

        # Execute operation based on arity (unary or binary)
        if opcode in ["not", "sig", "tanh", "relu"]:
            # Unary operations
            if b is not None:
                # If b is provided for unary operation, ignore it but log a warning
                import warnings

                warnings.warn(f"Second operand provided for unary operation {opcode}, it will be ignored")
            result = self.op_handlers[opcode](a, None)
        else:
            # Binary operations
            if b is None:
                raise ValueError(f"Second operand required for operation {opcode}")
            result = self.op_handlers[opcode](a, b)

        # Update statistics
        self._update_stats(operation_type, length)

        # Do clipping
        result = np.clip(result, self.min_value, self.max_value)

        return result

    def _handle_and(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return a & b

    def _handle_or(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return a | b

    def _handle_not(self, a: NDArray[np.int32], _) -> NDArray[np.int32]:
        return ~a

    def _handle_add(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return a + b

    def _handle_sub(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return a - b

    def _handle_mul(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return (a * b) >> self.frac_bits

    def _handle_div(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        if np.any(b == 0):
            raise ZeroDivisionError("Division by zero")
        return (a << self.frac_bits) // b

    def _handle_min(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return np.minimum(a, b)

    def _handle_max(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return np.maximum(a, b)

    def _handle_sig(self, a: NDArray[np.int32], _) -> NDArray[np.int32]:
        # Convert to float for tanh calculation
        float_input = a / (1 << self.frac_bits)
        # Calculate tanh
        float_result = 1 / (1 + np.exp(-float_input))
        # Scale back to int32
        int_result = (float_result * (1 << self.frac_bits)).astype(np.int32)
        return int_result

    def _handle_tanh(self, a: NDArray[np.int32], _) -> NDArray[np.int32]:
        # Convert to float for tanh calculation
        float_input = a / (1 << self.frac_bits)
        # Calculate tanh
        float_result = np.tanh(float_input)
        # Scale back to int32
        int_result = (float_result * (1 << self.frac_bits)).astype(np.int32)
        return int_result

    def _handle_relu(self, a: NDArray[np.int32], _) -> NDArray[np.int32]:
        return np.maximum(0, a)

    def _update_stats(self, operation_type: str, length: int) -> None:
        if operation_type == "mul":
            self.stats.mul_operations += length
        elif operation_type == "div":
            self.stats.div_operations += length
        elif operation_type == "act":
            self.stats.act_operations += length
        else:
            self.stats.other_operations += length
        self.stats.total_operations += length

    def get_stats(self) -> Stats:
        return self.stats.get_stats()
