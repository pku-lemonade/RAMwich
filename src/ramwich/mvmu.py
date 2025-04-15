from typing import List

import numpy as np

from .blocks.adc import ADCArray
from .blocks.dac import DACArray
from .blocks.xbar import XbarArray
from .config import ADCType, ADCConfig, Config, DACConfig, DataConfig, MVMUConfig, XBARConfig
from .stats import Stats
from .utils.data_convert import extract_bits, float_to_fixed, int_to_conductance


class MVMU:
    """
    Matrix-Vector Multiply unit with multiple crossbar arrays with detailed hardware simulation.
    """

    def __init__(self, id: int = 0, config: Config = None):
        # Basic MVMU properties
        self.id = id
        self.data_config = config.data_config or DataConfig()
        self.mvmu_config = config.mvmu_config or MVMUConfig()
        self.xbar_config = config.mvmu_config.xbar_config or XBARConfig()
        self.dac_config = config.mvmu_config.dac_config or DACConfig()
        self.adc_config = config.mvmu_config.adc_config or ADCConfig()

        # Initialize Xbar arrays
        self.rram_xbar_array = XbarArray(self.mvmu_config, self.data_config)

        # Initialize ADCs.
        # Each xbar has multiple ADCs based on the xbar_size divided by columns per ADC.
        # Number is multiplied by 2 for positive/negative crossbars for normal adcs, evens for positive and odds for negative.
        self.adc_array = ADCArray(self.mvmu_config, self.data_config)

        # Initialize DACs.
        # MVMU has multiple DACs based on the xbar_size, 1 DAC per column.
        # The same column of different xbars share the same DAC.
        # Positive and negative crossbars also share the same DAC.
        self.dac_array = DACArray(self.mvmu_config)

        # Memory components
        # self.xbar_memory = [[0.0 for _ in range(self.xbar_config.xbar_size)] for _ in range(self.mvmu_config.xbar_size)]
        # self.registers = [0 for _ in range(self.mvmu_config.num_registers)]

        # Initialize stats
        self.stats = Stats()

    def __repr__(self):
        return f"MVMU({self.id}, xbars={len(self.xbars)})"

    def load_weights(self, weights: List[float]):
        """Load weights into the crossbar arrays"""

        # Validate input length
        expected_length = self.xbar_config.xbar_size * self.xbar_config.xbar_size
        if len(weights) != expected_length:
            raise ValueError(
                f"Expected {expected_length} weight values for a {self.xbar_config.xbar_size}×{self.xbar_config.xbar_size} crossbar, but got {len(weights)}"
            )

        # Reshape to 2D matrix
        weights_matrix = np.array(weights).reshape(self.xbar_config.xbar_size, self.xbar_config.xbar_size)
        
        # Calculate signs of all weights at once
        signs = np.sign(weights_matrix)
        
        # Prepare weights with positive magnitudes
        abs_weights = np.abs(weights_matrix)
        
        # Convert all weights to fixed-point representation
        int_weights = np.vectorize(lambda w: float_to_fixed(w, self.data_config.frac_bits))(abs_weights)
        
        # Initialize the output array
        xbar_weights = np.zeros(
            (self.data_config.num_rram_xbar_per_matrix, self.xbar_config.xbar_size, self.xbar_config.xbar_size)
        )
        
        # Process each crossbar
        for k in range(self.data_config.num_rram_xbar_per_matrix):
            # Extract bits for this crossbar (still need to loop over k)
            xbar_int_weights = np.vectorize(
                lambda w: extract_bits(w, self.data_config.stored_bit[k], self.data_config.stored_bit[k + 1])
            )(int_weights)
            
            # Convert to conductance values (vectorized)
            conductance_values = np.vectorize(
                lambda w: int_to_conductance(
                    w,
                    self.data_config.bits_per_cell[k],
                    self.xbar_config.rram_conductance_min,
                    self.xbar_config.rram_conductance_max
                )
            )(xbar_int_weights)
            
            # Apply signs and store in result array
            xbar_weights[k] = signs * conductance_values

        # Load the processed weights into the xbar array
        self.rram_xbar_array.load_weights(xbar_weights)

    def _execute_mvm(self):
        """Execute a detailed matrix-vector multiplication instruction"""
        # dispatch DACs, crossbar, and ADCs
        pass

    def get_stats(self) -> Stats:
        return self.stats.get_stats(self.xbars + self.adcs + self.dacs)
