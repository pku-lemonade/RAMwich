import numpy as np
from numpy.typing import NDArray

from ..config import MVMUConfig
from ..stats import Stats

class SNAArray:
    """Hardware implementation of Shift and Add (SNA) array"""

    def __init__(self, mvmu_config: MVMUConfig=None):
        self.mvmu_config = mvmu_config or MVMUConfig()
        self.shape = (self.mvmu_config.num_xbar_per_mvmu, self.mvmu_config.num_adc_per_xbar)
        self.size = np.prod(self.shape)

        self.shift_bits = np.array(self.mvmu_config.stored_bit)[:, np.newaxis]

        # Initialize stats
        self.stats = Stats()

    def calculate(self, input_data: NDArray[np.integer]):
        """Performs the Shift and Add (SNA) operation on the input data
        
        Each SNA unit combines data from multiple xbars for one ADC position:
        - Takes the values from all xbars for that specific ADC
        - Applies appropriate bit shifts based on xbar position
        - Adds the shifted values to produce a single output
        
        Args:
            input_data: 2D array with shape (num_xbar_per_mvmu, num_adc_per_xbar)
            bits_per_cell: Array of bit shifts to apply for each xbar (optional)
            
        Returns:
            1D array with shape (num_adc_per_xbar,) containing SNA results
        """
        # Validate input data
        if input_data.shape != self.shape:
            raise ValueError(f"Input data shape {input_data.shape} does not match SNA array shape {self.shape}")
        
        # Apply shifts for each xbar
        shifted_data = input_data.astype(np.int64) << self.shift_bits # Prevent overflow during shifting
        
        # Sum across the xbar dimension to get final results
        result = np.sum(shifted_data, axis=0)
        
        # Update stats
        
        return result
