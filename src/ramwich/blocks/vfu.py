from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import Config, CoreConfig
from ..ops import VFUOpType
from ..stats import Stats, StatsDict


class VFUStats(BaseModel):
    """Statistics for VFU operations"""

    # Universal metrics
    config: CoreConfig = Field(default=CoreConfig(), description="Configuration object")

    # VFU specific metrics
    mul_operations: int = Field(default=0, description="Number of read operations")
    div_operations: int = Field(default=0, description="Number of write operations")
    act_operations: int = Field(default=0, description="Number of activation operations")
    other_operations: int = Field(default=0, description="Number of other operations")
    total_operations: int = Field(default=0, description="Total number of operations")

    def reset(self):
        """Reset all statistics to zero"""
        self.mul_operations = 0
        self.div_operations = 0
        self.act_operations = 0
        self.other_operations = 0
        self.total_operations = 0

    def get_stats(self) -> StatsDict:
        """Convert SRAMStats to general Stats object"""
        stats_dict = StatsDict()

        # Map VFU metrics to Stat object
        stats_dict["VFU"] = Stats(
            activation_count=self.total_operations,
            dynamic_energy=self.config.alu_pow_mul_dyn * self.mul_operations
            + self.config.alu_pow_div_dyn * self.div_operations
            + self.config.act_pow_dyn * self.act_operations
            + self.config.alu_pow_others_dyn * self.other_operations,
            leakage_energy=self.config.alu_pow_leak,
            area=self.config.alu_area + self.config.act_area,
        )
        stats_dict["VFU MUL"] = Stats(
            activation_count=self.mul_operations,
        )
        stats_dict["VFU DIV"] = Stats(
            activation_count=self.div_operations,
        )
        stats_dict["VFU ACT"] = Stats(
            activation_count=self.act_operations,
        )
        stats_dict["VFU OTHERS"] = Stats(
            activation_count=self.other_operations,
        )

        return stats_dict


class VFU:
    """SRAM register file component for the Core"""

    def __init__(self, config: Config):
        self.config = config
        self.frac_bits = self.config.data_config.activation_frac_bits
        self.min_value = -1 * (1 << self.config.data_config.activation_width)
        self.max_value = (1 << self.config.data_config.activation_width) - 1

        # Initialize operation handlers
        self._init_op_handlers()

        # Initialize stats
        self.stats = VFUStats(config=self.config.core_config)

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
        elif opcode in ["sig", "tanh"]:
            operation_type = "act"
        else:
            operation_type = "other"

        # Execute operation based on arity (unary or binary)
        if opcode in ["not", "sig", "tanh", "relu"]:
            # Unary operations
            if b is not None:
                # If b is provided for unary operation, ignore it but log a warning
                import warnings

                warnings.warn(f"Second operand provided for unary operation {opcode}, it will be ignored", stacklevel=2)
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

    def reset(self):
        """Reset all statistics"""
        self.stats.reset()

    def get_stats(self) -> StatsDict:
        return self.stats.get_stats()
