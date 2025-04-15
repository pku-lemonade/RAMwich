import numpy as np
from numpy.typing import NDArray

from ..config import MVMUConfig
from ..stats import Stats

class MuxArray:
    """Hardware implementation of a multiplexer array for selecting specific elements"""
    
    def __init__(self, mvmu_config: MVMUConfig=None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.num_input_per_mux = self.mvmu_config.num_columns_per_adc
        self.input_size = self.mvmu_config.num_rram_xbar_per_mvmu * self.mvmu_config.xbar_config.xbar_size
        self.size = self.input_size / self.num_input_per_mux
        
    def select(self, input_array: NDArray[np.floating], index: int):
        """Selects the value at the given index from the input array using a multiplexer"""
        
        # Validate input
        if input_array.ndim != 1:
            raise ValueError(f"Expected 1D array, got {input_array.ndim}D array")

        if len(input_array) != self.input_size:
            raise ValueError(f"Expected input array of length {self.input_size}, got {len(input_array)}")
        
        if index < 0 or index >= self.num_input_per_mux:
            raise ValueError(f"Index {index} out of bounds")
        
        # Select the value at the given index
        indices = np.arange(index, len(input_array), self.num_input_per_mux)

        return input_array[indices]