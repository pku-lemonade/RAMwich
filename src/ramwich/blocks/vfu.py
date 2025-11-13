from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ..config import Config, CoreConfig
from ..ops import FVFUOpType, IVFUOpType
from ..stats import Stats, StatsDict


class FVFUStats(BaseModel):
    """Statistics for Floating Point VFU operations"""

    # Universal metrics
    config: CoreConfig = Field(default=CoreConfig(), description="Configuration object")

    # FVFU specific metrics
    mul_operations: int = Field(default=0, description="Number of multiplication operations")
    div_operations: int = Field(default=0, description="Number of division operations")
    act_operations: int = Field(default=0, description="Number of activation operations")
    convert_operations: int = Field(default=0, description="Number of conversion operations")
    other_operations: int = Field(default=0, description="Number of other operations")
    total_operations: int = Field(default=0, description="Total number of operations")

    def reset(self):
        """Reset all statistics to zero"""
        self.mul_operations = 0
        self.div_operations = 0
        self.act_operations = 0
        self.convert_operations = 0
        self.other_operations = 0
        self.total_operations = 0

    def get_stats(self) -> StatsDict:
        """Convert FVFUStats to general Stats object"""
        stats_dict = StatsDict()

        # Map VFU metrics to Stat object
        stats_dict["FVFU"] = Stats(
            activation_count=self.total_operations,
            dynamic_energy=self.config.fvfu_pow_mul_dyn * self.mul_operations
            + self.config.fvfu_pow_div_dyn * self.div_operations
            + self.config.act_pow_dyn * self.act_operations
            + self.config.fvfu_pow_convert_dyn * self.convert_operations
            + self.config.fvfu_pow_others_dyn * self.other_operations,
            leakage_energy=self.config.fvfu_pow_leak,
            area=self.config.fvfu_area + self.config.act_area,
        )

        return stats_dict


class IVFUStats(BaseModel):
    """Statistics for Integer VFU operations"""

    # Universal metrics
    config: CoreConfig = Field(default=CoreConfig(), description="Configuration object")

    # Integer VFU specific metrics
    mul_operations: int = Field(default=0, description="Number of multiplication operations")
    div_operations: int = Field(default=0, description="Number of division operations")
    other_operations: int = Field(default=0, description="Number of other operations")
    total_operations: int = Field(default=0, description="Total number of operations")

    def reset(self):
        """Reset all statistics to zero"""
        self.mul_operations = 0
        self.div_operations = 0
        self.other_operations = 0
        self.total_operations = 0

    def get_stats(self) -> StatsDict:
        """Convert Integer VFU Stats to general Stats object"""
        stats_dict = StatsDict()

        # Map VFU metrics to Stat object
        stats_dict["IVFU"] = Stats(
            activation_count=self.total_operations,
            dynamic_energy=self.config.alu_pow_mul_dyn * self.mul_operations
            + self.config.alu_pow_div_dyn * self.div_operations
            + self.config.alu_pow_others_dyn * self.other_operations,
            leakage_energy=self.config.alu_pow_leak,
            area=self.config.alu_area + self.config.act_area,
        )

        return stats_dict


class FVFU:
    """Floating Point Vector Functional Unit (VFU) for the Core"""

    def __init__(self, config: Config):
        # Initialize configuration
        self.config = config

        # Initialize operation handlers
        self._init_op_handlers()

        # Initialize stats
        self.stats = FVFUStats(config=self.config.core_config)

    def _init_op_handlers(self):
        """Initialize operation handlers dictionary"""
        self.op_handlers = {
            "add": self._handle_add,
            "addi": self._handle_addi,
            "sub": self._handle_sub,
            "subi": self._handle_subi,
            "mul": self._handle_mul,
            "muli": self._handle_muli,
            "div": self._handle_div,
            "divi": self._handle_divi,
            "min": self._handle_min,
            "max": self._handle_max,
            "sig": self._handle_sig,
            "tanh": self._handle_tanh,
            "relu": self._handle_relu,
            "int_to_fp": self._handle_int_to_fp,
            "fp_to_int": self._handle_fp_to_int,
        }

    def calculate(
        self,
        opcode: FVFUOpType,
        a: Union[NDArray[np.float32], NDArray[np.int32], np.float32, np.int32],
        b: Optional[Union[NDArray[np.float32], np.float32]] = None,
        imm: Optional[np.float32] = None,
    ) -> Union[NDArray[np.float32], NDArray[np.int32]]:
        """Perform a calculation using the ALU and activation unit.

        Notes on input and output dtypes:
        - Most floating-point ops (add/sub/mul/div/min/max/sig/tanh/relu) take FP32 inputs and return FP32 arrays.
        - Conversion ops:
          * int_to_fp: accepts INT32 input and returns FP32 output.
          * fp_to_int: accepts FP32 input and returns INT32 output.
        """

        if isinstance(a, (int, np.int32)):
            # Normalize scalar to 1D array with appropriate dtype based on opcode
            if opcode == "int_to_fp":
                a = np.array([a], dtype=np.int32)
            else:
                raise ValueError("Scalar integer input is only supported for int_to_fp operation")
        elif isinstance(a, float):
            a = np.array([a], dtype=np.float32)

        length = len(a)

        if b is not None and isinstance(b, (float, np.float32)):
            # Secondary operand is only used for binary FP ops; keep as FP32
            b = np.array([b], dtype=np.float32)
            if len(b) != length:
                raise ValueError("Operands must be of the same length")

        # Determine operation type for statistics
        if opcode in ["mul", "div"]:
            operation_type = opcode
        elif opcode in ["sig", "tanh"]:
            operation_type = "act"
        elif opcode in ["int_to_fp", "fp_to_int"]:
            operation_type = "convert"
        else:
            operation_type = "other"

        # Execute operation based on arity (unary or binary or immediate)
        if opcode in ["not", "sig", "tanh", "relu", "int_to_fp", "fp_to_int"]:
            # Unary operations
            if b is not None:
                # If b is provided for unary operation, ignore it but log a warning
                import warnings

                warnings.warn(f"Second operand provided for unary operation {opcode}, it will be ignored", stacklevel=2)
            result = self.op_handlers[opcode](a, None, None)
        elif opcode in ["addi", "subi", "muli", "divi"]:
            # Immediate operations
            assert imm is not None, f"Immediate value must be provided for operation {opcode}"
            if b is not None:
                # If b is provided for immediate operation, ignore it but log a warning
                import warnings

                warnings.warn(
                    f"Second operand provided for immediate operation {opcode}, it will be ignored", stacklevel=2
                )
            result = self.op_handlers[opcode](a, None, imm)
        else:
            # Binary operations
            if b is None:
                raise ValueError(f"Second operand required for operation {opcode}")
            result = self.op_handlers[opcode](a, b, imm)

        # Update statistics
        self._update_stats(operation_type, length)

        return result

    def _handle_add(self, a: NDArray[np.float32], b: NDArray[np.float32], _) -> NDArray[np.float32]:
        return a + b

    def _handle_addi(self, a: NDArray[np.float32], _, imm: np.float32) -> NDArray[np.float32]:
        return a + imm

    def _handle_sub(self, a: NDArray[np.float32], b: NDArray[np.float32], _) -> NDArray[np.float32]:
        return a - b

    def _handle_subi(self, a: NDArray[np.float32], _, imm: np.float32) -> NDArray[np.float32]:
        return a - imm

    def _handle_mul(self, a: NDArray[np.float32], b: NDArray[np.float32], _) -> NDArray[np.float32]:
        return a * b

    def _handle_muli(self, a: NDArray[np.float32], _, imm: np.float32) -> NDArray[np.float32]:
        return a * imm

    def _handle_div(self, a: NDArray[np.float32], b: NDArray[np.float32], _) -> NDArray[np.float32]:
        if np.any(b == 0):
            raise ZeroDivisionError("Division by zero")
        return a / b

    def _handle_divi(self, a: NDArray[np.float32], _, imm: np.float32) -> NDArray[np.float32]:
        if imm == 0:
            raise ZeroDivisionError("Division by zero")
        return a / imm

    def _handle_min(self, a: NDArray[np.float32], b: NDArray[np.float32], _) -> NDArray[np.float32]:
        return np.minimum(a, b)

    def _handle_max(self, a: NDArray[np.float32], b: NDArray[np.float32], _) -> NDArray[np.float32]:
        return np.maximum(a, b)

    def _handle_sig(self, a: NDArray[np.float32], _, __) -> NDArray[np.float32]:
        # Calculate sigmoid
        return 1 / (1 + np.exp(-a))

    def _handle_tanh(self, a: NDArray[np.float32], _, __) -> NDArray[np.float32]:
        # Calculate tanh
        return np.tanh(a)

    def _handle_relu(self, a: NDArray[np.float32], _, __) -> NDArray[np.float32]:
        return np.maximum(0, a)

    def _handle_int_to_fp(self, a: NDArray[np.int32], _, __) -> NDArray[np.float32]:
        return a.astype(np.float32)

    def _handle_fp_to_int(self, a: NDArray[np.float32], _, __) -> NDArray[np.int32]:
        result = np.clip(np.round(a), self.config.core_config.int_min_value, self.config.core_config.int_max_value)
        return result.astype(np.int32)

    def _update_stats(self, operation_type: str, length: int) -> None:
        if operation_type == "mul":
            self.stats.mul_operations += length
        elif operation_type == "div":
            self.stats.div_operations += length
        elif operation_type == "act":
            self.stats.act_operations += length
        elif operation_type == "convert":
            self.stats.convert_operations += length
        else:
            self.stats.other_operations += length
        self.stats.total_operations += length

    def reset(self):
        """Reset all statistics"""
        self.stats.reset()

    def get_stats(self) -> StatsDict:
        return self.stats.get_stats()


class IVFU:
    """Integer Vector Functional Unit (VFU) for the Core"""

    def __init__(self, config: Config):
        # Initialize configuration
        self.config = config

        # Initialize operation handlers
        self._init_op_handlers()

        # Initialize stats
        self.stats = IVFUStats(config=self.config.core_config)

    def _init_op_handlers(self):
        """Initialize operation handlers dictionary"""
        self.op_handlers = {
            "and": self._handle_and,
            "or": self._handle_or,
            "xor": self._handle_xor,
            "not": self._handle_not,
            "add": self._handle_add,
            "sub": self._handle_sub,
            "mul": self._handle_mul,
            "div": self._handle_div,
            "min": self._handle_min,
            "max": self._handle_max,
        }

    def calculate(
        self, opcode: IVFUOpType, a: Union[NDArray[np.int32], int], b: Optional[Union[NDArray[np.int32], int]] = None
    ) -> NDArray[np.int32]:
        """Perform a calculation using ALU"""
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
        else:
            operation_type = "other"

        # Execute operation based on arity (unary or binary)
        if opcode == "not":
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

        return result

    def _handle_and(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return a & b

    def _handle_or(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return a | b

    def _handle_xor(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return a ^ b

    def _handle_not(self, a: NDArray[np.int32], _) -> NDArray[np.int32]:
        return ~a

    def _handle_add(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return a + b

    def _handle_sub(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return a - b

    def _handle_mul(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return a * b

    def _handle_div(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        if np.any(b == 0):
            raise ZeroDivisionError("Division by zero")
        return a // b

    def _handle_min(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return np.minimum(a, b)

    def _handle_max(self, a: NDArray[np.int32], b: NDArray[np.int32]) -> NDArray[np.int32]:
        return np.maximum(a, b)

    def _update_stats(self, operation_type: str, length: int) -> None:
        if operation_type == "mul":
            self.stats.mul_operations += length
        elif operation_type == "div":
            self.stats.div_operations += length
        else:
            self.stats.other_operations += length
        self.stats.total_operations += length

    def reset(self):
        """Reset all statistics"""
        self.stats.reset()

    def get_stats(self) -> StatsDict:
        return self.stats.get_stats()
