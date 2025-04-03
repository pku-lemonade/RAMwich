from typing import Literal, Union, List, Optional
from pydantic import BaseModel, Field
from .core import Core
from .tile import Tile

class BaseOp(BaseModel):
    node: int
    tile: int

class CoreOp(BaseOp):
    core: int

    def accept(self, core: Core):
        pass

class TileOp(BaseOp):
    def accept(self, tile: Tile):
        pass

class Load(CoreOp):
    type: Literal["load"] = "load"
    imm: int  # Immediate value to load
    d1: int  # Memory address to load from

    def accept(self, core):
        return core.execute_load(self.d1)

class Set(CoreOp):
    type: Literal["set"] = "set"
    imm: int  # Immediate value to store

    def accept(self, core):
        return core.execute_set(self.imm)

class ALU(CoreOp):
    type: Literal["alu"] = "alu"
    opcode: str

    def accept(self, core):
        return core.execute_alu(self.opcode)

class MVM(CoreOp):
    type: Literal["mvm"] = "mvm"
    xbar: List[int]
    ima: int = 0  # Default IMA id

    def accept(self, core):
        return core.execute_mvm(self.ima, self.xbar)

# Add explicit memory operations if needed
class Store(CoreOp):
    type: Literal["store"] = "store"
    address: int  # Memory address to store to
    value: int    # Value to store

    def accept(self, core):
        return core.execute_store(self.address, self.value)

class Send(TileOp):
    type: Literal["send"] = "send"
    mem_addr: int     # Memory address to send from
    vtile_id: int     # Virtual tile ID to send to
    width: int        # Width of data to send
    counter: int      # Counter for tracking sends
    vec: List[int]    # Vector of data to send

    def accept(self, tile):
        return tile.execute_send(self.mem_addr, self.vtile_id, self.width, self.counter, self.vec)

class Recv(TileOp):
    type: Literal["receive"] = "receive"
    mem_addr: int     # Memory address to receive into
    vtile_id: int     # Virtual tile ID to receive from
    width: int        # Width of data to receive
    counter: int      # Counter for tracking receives
    vec: List[int]    # Vector to store received data

    def accept(self, tile):
        return tile.execute_receive(self.mem_addr, self.vtile_id, self.width, self.counter, self.vec)

# Create discriminated union types
CoreOpType = Union[Load, Set, ALU, MVM, Store]
TileOpType = Union[Send, Recv]
OpType = Union[CoreOpType, TileOpType]
