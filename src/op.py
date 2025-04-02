from pydantic import BaseModel

class Op(BaseModel):
    tile: int
    core: int

class Load(Op):
    d1: int

class Set(Op):
    imm: int

class Alu(Op):
    opcode: str

class MVM(Op):
    xbar: list[int]
