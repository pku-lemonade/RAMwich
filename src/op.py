from pydantic import BaseModel

class Op(BaseModel):
    tile: int
    core: int
    node: int = 0  # Default to node 0

    def accept(self, core):
        pass

class Load(Op):
    d1: int

    def accept(self, core):
        return core.execute_load(self.d1)

class Set(Op):
    imm: int

    def accept(self, core):
        return core.execute_set(self.imm)

class Alu(Op):
    opcode: str

    def accept(self, core):
        return core.execute_alu(self.opcode)

class MVM(Op):
    xbar: list[int]
    ima: int = 0  # Default IMA id

    def accept(self, core):
        return core.execute_mvm(self.ima, self.xbar)
