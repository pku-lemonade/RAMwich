from typing import List

import numpy as np

from .blocks.adc import ADC
from .blocks.dac import DAC
from .blocks.xbar import Xbar
from .config import DataConfig, DACConfig, XBARConfig, ADCConfig, MVMUConfig, Config
from .stats import Stats
from .utils.data_convert import int2conductance


class MVMU:
    """
    Matrix-Vector Multiply unit with multiple crossbar arrays with detailed hardware simulation.
    """

    def __init__(self, id: int = 0, config: Config = None):
        # Basic MVMU properties
        self.id = id
        self.data_config = config.data_config or DataConfig()
        self.dac_config = config.dac_config or DACConfig()
        self.xbar_config = config.xbar_config or XBARConfig()
        self.adc_config = config.adc_config or ADCConfig()
        self.mvmu_config = config.mvmu_config or MVMUConfig()

        # Initialize Xbar arrays
        self.xbars = [
            Xbar(i, self.xbar_config.xbar_size if hasattr(self.mvmu_config, "xbar_size") else 32)
            for i in range(self.data_config.reram_xbar_num_per_matrix)
        ]

        self.stats = Stats()

        # Initialize sub-components
        self.adcs = [ADC(self.adc_config) for _ in range(int(self.xbar_config.xbar_size // self.mvmu_config.num_columns_per_adc * 2))]
        self.dacs = [DAC(self.dac_config) for _ in range(self.xbar_config.xbar_size)]

        # Memory components
        #self.xbar_memory = [[0.0 for _ in range(self.xbar_config.xbar_size)] for _ in range(self.mvmu_config.xbar_size)]
        #self.registers = [0 for _ in range(self.mvmu_config.num_registers)]

    def __repr__(self):
        return f"MVMU({self.id}, xbars={len(self.xbars)})"
    
    def load_weights(self, values: List[float]):
        """Load weights into the crossbar arrays"""

        # Validate input length
        expected_length = self.xbar_config.xbar_size * self.xbar_config.xbar_size
        if len(values) != expected_length:
            raise ValueError(f"Expected {expected_length} weight values for a {self.xbar_config.xbar_size}×{self.xbar_config.xbar_size} crossbar, but got {len(values)}")

        logical_xbar = np.array(values).reshape(self.xbar_config.xbar_size, self.xbar_config.xbar_size)
        phy_xbar = np.zeros((self.data_config.reram_xbar_num_per_matrix, self.xbar_config.xbar_size, self.xbar_config.xbar_size))

        for i in range(self.xbar_config.xbar_size):
            for j in range(self.xbar_config.xbar_size):
                sign = 1 if logical_xbar[i][j] >= 0 else -1 # mark if we are storing a negative number, positive and negative are stored separately
                int_val = int(sign * logical_xbar[i][j] * (2 ** self.data_config.frac_bits))
                    
                for k in range(self.data_config.reram_xbar_num_per_matrix):
                    clipped_val = int_val >> (self.data_config.num_bits >> self.data_config.stored_bit[k]) & ((1 << self.data_config.bits_per_cell[k]) - 1)

                    # we storage negative resistance values here.
                    # when programing to xbar it will be separated to a positive xbar and a negative xbar
                    phy_xbar[k][i][j] = sign * int2conductance(clipped_val, self.data_config.bits_per_cell[k], self.xbar_config.reram_conductance_min, self.xbar_config.reram_conductance_max)
        
        for i in range(self.data_config.reram_xbar_num_per_matrix):
            self.xbars[i].load_weights(phy_xbar[i])

    def _execute_mvm(self, instruction):
        """Execute a detailed matrix-vector multiplication instruction"""
        # dispatch DACs, crossbar, and ADCs
        pass

    def get_stats(self) -> Stats:
        return self.stats.get_stats(self.xbars + self.adcs + self.dacs)
