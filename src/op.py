from typing import Literal, Union, List
from pydantic import BaseModel, Field

class Op(BaseModel):
    node: int
    tile: int
    core: int

    def accept(self, core):
        pass

class Load(Op):
    type: Literal["load"]
    d1: int  # Memory address to load from

    def accept(self, core):
        return core.execute_load(self.d1)

class Set(Op):
    type: Literal["set"] = "set"
    imm: int  # Immediate value to store

    def accept(self, core):
        return core.execute_set(self.imm)

class ALU(Op):
    type: Literal["alu"] = "alu"
    opcode: str

    def accept(self, core):
        return core.execute_alu(self.opcode)

class MVM(Op):
    type: Literal["mvm"] = "mvm"
    xbar: List[int]
    ima: int = 0  # Default IMA id

    def accept(self, core):
        return core.execute_mvm(self.ima, self.xbar)

# Add explicit memory operations if needed
class Store(Op):
    type: Literal["store"] = "store"
    address: int  # Memory address to store to
    value: int    # Value to store

    def accept(self, core):
        try:
            return core.dram.write(self.address, self.value)
        except IndexError:
            return False

# Create a discriminated union type for operations
OpType = Union[Load, Set, ALU, MVM, Store]
