from abc import ABC, abstractmethod
from typing import Literal, Union, List, Optional
from pydantic import BaseModel, Field

class BaseOp(BaseModel, ABC):
    type: str
    node: int

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
    imm: int  # Immediate value to load
    d1: int  # Memory address to load from

    def accept(self, visitor):
        return visitor.visit_load(self)

class Set(CoreOp):
    type: Literal["set"] = "set"
    imm: int  # Immediate value to store

    def accept(self, visitor):
        return visitor.visit_set(self)

class ALU(CoreOp):
    type: Literal["alu"] = "alu"
    opcode: str

    def accept(self, visitor):
        return visitor.visit_alu(self)

class MVM(CoreOp):
    type: Literal["mvm"] = "mvm"
    xbar: List[int]
    ima: int = 0  # Default IMA id

    def accept(self, visitor):
        return visitor.visit_mvm(self)

# Add explicit memory operations if needed
class Store(CoreOp):
    type: Literal["store"] = "store"
    address: int  # Memory address to store to
    value: int    # Value to store

    def accept(self, visitor):
        return visitor.visit_store(self)

class Send(TileOp):
    type: Literal["send"] = "send"
    mem_addr: int     # Memory address to send from
    vtile_id: int     # Virtual tile ID to send to
    width: int        # Width of data to send
    counter: int      # Counter for tracking sends
    vec: List[int]    # Vector of data to send

    def accept(self, tile):
        return tile.execute_send(self)

class Recv(TileOp):
    type: Literal["receive"] = "receive"
    mem_addr: int     # Memory address to receive into
    vtile_id: int     # Virtual tile ID to receive from
    width: int        # Width of data to receive
    counter: int      # Counter for tracking receives
    vec: List[int]    # Vector to store received data

    def accept(self, tile):
        return tile.execute_receive(self)

# Removed TimingVisitor and ExecutionVisitor classes as they are now in core.py

# Create discriminated union types
CoreOpType = Union[Load, Set, ALU, MVM, Store]
TileOpType = Union[Send, Recv]
OpType = Union[CoreOpType, TileOpType]
