from typing import List

class Xbar:
    """
    Crossbar array in the RAMwich architecture.
    """
    def __init__(self, id: int, size: int = 32):
        self.id = id
        self.size = size
        self.memory = [0] * size

    def __repr__(self):
        return f"Xbar({self.id}, size={self.size})"

    def set_values(self, values: List[int]):
        """Set values in the crossbar"""
        for i, val in enumerate(values):
            if i < self.size:
                self.memory[i] = val
            else:
                break

class IMA:
    """
    In-Memory Accelerator containing multiple crossbar arrays.
    """
    def __init__(self, id: int, num_xbars: int = 4):
        self.id = id
        self.xbars = [Xbar(i) for i in range(num_xbars)]

    def __repr__(self):
        return f"IMA({self.id}, xbars={len(self.xbars)})"

    def execute_mvm(self, xbar_values: List[int]):
        """Execute a Matrix-Vector Multiplication operation"""
        # Placeholder for actual implementation
        # Typically would perform matrix operations using the crossbar arrays

        # For now, just store values in the first xbar
        if self.xbars:
            self.xbars[0].set_values(xbar_values)

        return True
