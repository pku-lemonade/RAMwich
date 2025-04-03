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
    d1: int

    def accept(self, core):
        return core.execute_load(self.d1)

class Set(Op):
    type: Literal["set"] = "set"
    imm: int

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

# Create a discriminated union type for operations
OpType = Union[Load, Set, ALU, MVM]
