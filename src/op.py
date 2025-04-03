from pydantic import BaseModel

class Op(BaseModel):
    tile: int
    core: int
    node: int = 0  # Default to node 0

    def accept(self, simulator):
        pass

class Load(Op):
    d1: int

    def accept(self, simulator):
        return simulator.execute_load(self)

class Set(Op):
    imm: int

    def accept(self, simulator):
        return simulator.execute_set(self)

class Alu(Op):
    opcode: str

    def accept(self, simulator):
        return simulator.execute_alu(self)

class MVM(Op):
    xbar: list[int]
    ima: int = 0  # Default IMA id

    def accept(self, simulator):
        return simulator.execute_mvm(self)
