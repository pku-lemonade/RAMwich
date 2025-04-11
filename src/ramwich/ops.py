from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


class BaseOp(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)
    type: str
    node: int = 0

    @abstractmethod
    def accept(self, visitor):
        raise NotImplementedError


class CoreOp(BaseOp):
    tile: int
    core: int


class TileOp(BaseOp):
    tile: int


class Load(CoreOp):
    type: Literal["load"] = "load"
    dest: int  # Target register where loaded data will be stored
    read: int  # Register containing the memory address to read from (indirect addressing)
    width: int  # Number of values to load in each operation (based on bus width)
    vec: int  # Total number of load operations needed to complete this vector load

    def accept(self, visitor):
        return visitor.visit_load(self)


class Store(CoreOp):
    type: Literal["store"] = "store"
    dest: int  # Register containing the memory address to write to (indirect addressing)
    read: int  # Register to read from
    width: int  # Number of values to store in each operation (based on bus width)
    vec: int  # Total number of store operations needed to complete this vector store

    def accept(self, visitor):
        return visitor.visit_store(self)


class Set(CoreOp):
    type: Literal["set"] = "set"
    dest: int  # Target register where values will be stored
    imm: int  # Immediate value to store
    vec: int  # Length of vector (all elements will be the imm)
    is_address: bool  # Marks that if this set operation stores an address or normal data

    def accept(self, visitor):
        return visitor.visit_set(self)


class Copy(CoreOp):
    type: Literal["copy"] = "copy"
    dest: int  # Target register where datas will be stored
    read: int  # Register containing the datas to copy
    vec: int  # Length of vector

    def accept(self, visitor):
        return visitor.visit_copy(self)


class MVM(CoreOp):
    type: Literal["mvm"] = "mvm"
    xbar: List[int]  # The MVMUs to be activated

    def accept(self, visitor):
        return visitor.visit_mvm(self)


class ALU(CoreOp):
    type: Literal["alu"] = "alu"
    opcode: Literal["add", "sub", "mul", "div", "max", "sig", "tanh", "relu"]
    dest: int  # Target register where the result vector will be stored
    read_1: int  # Register that stores operator 1
    read_2: Optional[int] = None  # Register that stores operator 2, Not needed in sig, tanh and relu
    imm: Optional[int] = None  # Immediate value to use some operation
    vec: int  # Length of vector

    def accept(self, visitor):
        return visitor.visit_alu(self)


class Hlt(CoreOp):
    type: Literal["hlt"] = "hlt"

    def accept(self, visitor):
        return visitor.visit_hlt(self)


class Send(TileOp):
    type: Literal["send"] = "send"
    mem_addr: int  # Memory address to read from
    target_tile: int  # Virtual tile ID to send to
    width: int  # Number of values to send in each operation (based on noc bus width)
    vec: int  # Total number of send operations needed to complete this vector send

    def accept(self, tile):
        return tile.execute_send(self)


class Recv(TileOp):
    type: Literal["receive"] = "receive"
    mem_addr: int  # Memory address to write to
    source_tile: int  # Virtual tile ID to receive from
    width: int  # Number of values to Receive in each operation (based on noc bus width)
    vec: int  # Total number of receive operations needed to complete this vector receive

    def accept(self, tile):
        return tile.execute_receive(self)


class Halt(TileOp):
    type: Literal["halt"] = "halt"

    def accept(self, tile):
        return tile.execute_halt(self)


# Removed TimingVisitor and ExecutionVisitor classes as they are now in core.py

# Create discriminated union types
CoreOpType = Union[Load, Store, Set, Copy, MVM, ALU, Hlt]
TileOpType = Union[Send, Recv, Halt]
OpType = Union[CoreOpType, TileOpType]


class Operation(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: OpType


class Weight(BaseModel):
    model_config = ConfigDict(frozen=True)

    node: int = 0
    tile: int
    core: int
    mvmu: int
    value: List[float]
