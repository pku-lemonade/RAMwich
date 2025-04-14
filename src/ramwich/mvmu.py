from typing import List

import numpy as np
import sys

from .blocks.adc import ADC
from .blocks.dac import DAC
from .blocks.xbar import Xbar
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
        self.dac_config = config.dac_config or DACConfig()
        self.xbar_config = config.xbar_config or XBARConfig()
        self.adc_config_list = []
        for i in range(self.data_config.num_rram_xbar_per_matrix):
            temp_adc_config = config.adc_config or ADCConfig()
            conductance_step = (
                (self.xbar_config.rram_conductance_max - self.xbar_config.rram_conductance_min) /
                (2 ** self.data_config.bits_per_cell[i] - 1)
            )
            voltage_step = self.dac_config.VDD / (2 ** self.dac_config.resolution - 1)
            temp_adc_config.step = voltage_step * conductance_step
            self.adc_config_list.append(temp_adc_config)
        self.mvmu_config = config.mvmu_config or MVMUConfig()

        # Initialize Xbar arrays
        self.xbars = [
            Xbar(i, self.xbar_config.xbar_size if hasattr(self.mvmu_config, "xbar_size") else 32)
            for i in range(self.data_config.num_rram_xbar_per_matrix)
        ]

        self.stats = Stats()

        # Initialize 2D array of ADCs organized as adcs[xbar_id][adc_id].
        # Each xbar has multiple ADCs based on the xbar_size divided by columns per ADC.
        # Number is multiplied by 2 for positive/negative crossbars for normal adcs, evens for positive and odds for negative.
        adc_num_factor = 1 if self.adc_config_list[0].type == ADCType.DIFFERENTIAL else 2
        self.adcs = [[
                ADC(self.adc_config_list[i])
                for _ in range(int(self.xbar_config.xbar_size // self.mvmu_config.num_columns_per_adc * adc_num_factor))
            ]
            for i in range(self.data_config.num_rram_xbar_per_matrix)
        ]
        # Initialize DACs.
        # MVMU has multiple DACs based on the xbar_size, 1 DAC per column.
        # The same column of different xbars share the same DAC.
        # Positive and negative crossbars also share the same DAC.
        self.dacs = [DAC(self.dac_config) for _ in range(self.xbar_config.xbar_size)]

        # Memory components
        # self.xbar_memory = [[0.0 for _ in range(self.xbar_config.xbar_size)] for _ in range(self.mvmu_config.xbar_size)]
        # self.registers = [0 for _ in range(self.mvmu_config.num_registers)]

    def __repr__(self):
        return f"MVMU({self.id}, xbars={len(self.xbars)})"

    def load_weights(self, weights: List[float]):
        """Load weights into the crossbar arrays"""

        # Validate input length
        expected_length = self.xbar_config.xbar_size * self.xbar_config.xbar_size
        if len(weights) != expected_length:
            raise ValueError(
                f"Expected {expected_length} weight values for a {self.xbar_config.xbar_size}×{self.xbar_config.xbar_size} crossbar, but got {len(values)}"
            )

        weights = np.array(weights).reshape(self.xbar_config.xbar_size, self.xbar_config.xbar_size)
        xbar_weights = np.zeros(
            (self.data_config.num_rram_xbar_per_matrix, self.xbar_config.xbar_size, self.xbar_config.xbar_size)
        )

        for i in range(self.xbar_config.xbar_size):
            for j in range(self.xbar_config.xbar_size):
                sign = 1 if weights[i][j] >= 0 else -1
                int_weight = float_to_fixed(sign * weights[i][j], self.data_config.frac_bits)
                for k in range(self.data_config.num_rram_xbar_per_matrix):
                    xbar_int_weight = extract_bits(
                        int_weight, self.data_config.stored_bit[k], self.data_config.stored_bit[k + 1]
                    )
                    # Here we storage negative resistance values to physical xbar.
                    # When programing to xbar it will be separated to a positive xbar and a negative xbar
                    xbar_weights[k][i][j] = sign * int_to_conductance(
                        xbar_int_weight,
                        self.data_config.bits_per_cell[k],
                        self.xbar_config.rram_conductance_min,
                        self.xbar_config.rram_conductance_max,
                    )

        for k in range(self.data_config.num_rram_xbar_per_matrix):
            self.xbars[k].load_weights(xbar_weights[k])

    def _execute_mvm(self, instruction):
        """Execute a detailed matrix-vector multiplication instruction"""
        # dispatch DACs, crossbar, and ADCs
        pass

    def get_stats(self) -> Stats:
        return self.stats.get_stats(self.xbars + self.adcs + self.dacs)
