import numpy as np
from numpy.typing import NDArray

from ..config import MVMUConfig
from ..stats import Stats

class MuxArray:
    """Hardware implementation of a multiplexer array for selecting specific elements"""
    
    def __init__(self, mvmu_config: MVMUConfig=None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.num_xbar = self.mvmu_config.num_rram_xbar_per_mvmu
        self.num_input_per_mux = self.mvmu_config.num_columns_per_adc
        self.num_mux_per_xbar = self.mvmu_config.xbar_config.xbar_size // self.num_input_per_mux
        
        # Define 2D shape

        self.input_shape = (self.num_xbar, self.mvmu_config.xbar_config.xbar_size)
        self.output_shape = (self.num_xbar, self.num_mux_per_xbar)
        self.size = np.prod(self.output_shape)

        # Initialize stats
        self.stats = Stats()
        
    def select(self, input_array: NDArray[np.floating], index: int):
        """Selects the value at the given index from the input array using a multiplexer"""
        
        # Validate input
        if input_array.shape != self.input_shape:
            raise ValueError(f"Expected input array of shape {self.input_shape}, got {input_array.shape}")
        
        if index < 0 or index >= self.num_input_per_mux:
            raise ValueError(f"Index {index} out of bounds")
        
        reshaped_input = input_array.reshape(self.num_xbar, self.num_mux_per_xbar, self.num_input_per_mux)

        return reshaped_input[:, :, index]