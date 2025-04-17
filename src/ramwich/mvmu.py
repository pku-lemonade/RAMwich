from typing import List

import numpy as np

from .blocks.adc import ADCArray
from .blocks.dac import DACArray
from .blocks.inreg import InputRegisterArray
from .blocks.mux import MuxArray
from .blocks.outreg import OutputRegisterArray
from .blocks.sna import SNAArray
from .blocks.snh import SNHArray
from .blocks.xbar import XbarArray
from .config import (
    ADCConfig,
    ADCType,
    Config,
    DACConfig,
    DataConfig,
    MVMUConfig,
    XBARConfig,
)
from .stats import Stats
from .utils.data_convert import extract_bits, float_to_fixed, int_to_conductance


class MVMU:
    """
    Matrix-Vector Multiply unit with multiple crossbar arrays with detailed hardware simulation.
    """

    def __init__(self, id: int = 0, config: Config = None):
        # Basic MVMU properties
        self.id = id
        self.config = config or Config()
        self.data_config = self.config.data_config
        self.mvmu_config = self.config.mvmu_config

        # Initialize Xbar arrays
        self.rram_xbar_array = XbarArray(self.mvmu_config)

        # Initialize ADCs.
        # Each xbar has multiple ADCs based on the xbar_size divided by columns per ADC.
        # Number is multiplied by 2 for positive/negative crossbars for normal adcs, evens for positive and odds for negative.
        self.adc_array = ADCArray(self.mvmu_config)

        # Initialize DACs.
        # MVMU has multiple DACs based on the xbar_size, 1 DAC per column.
        # The same column of different xbars share the same DAC.
        # Positive and negative crossbars also share the same DAC.
        self.dac_array = DACArray(self.mvmu_config)

        # Memory components
        self.input_register_array = InputRegisterArray(self.mvmu_config)
        self.output_register_array = OutputRegisterArray(self.mvmu_config)

        # Initialize other components
        self.snh_array_pos = SNHArray(self.mvmu_config)
        self.snh_array_neg = SNHArray(self.mvmu_config)
        self.mux_array_pos = MuxArray(self.mvmu_config)
        self.mux_array_neg = MuxArray(self.mvmu_config)
        self.sna_array = SNAArray(self.mvmu_config)

        # Initialize stats
        self.stats = Stats()

    def __repr__(self):
        return f"MVMU({self.id}, xbars={len(self.xbars)})"

    def load_weights(self, weights: List[float]):
        """Load weights into the crossbar arrays"""

        # Validate input length
        xbar_size = self.mvmu_config.xbar_config.xbar_size
        expected_length = xbar_size * xbar_size
        if len(weights) != expected_length:
            raise ValueError(
                f"Expected {expected_length} weight values for a {xbar_size}×{xbar_size} crossbar, but got {len(weights)}"
            )

        # Reshape to 2D matrix
        weights_matrix = np.array(weights).reshape(xbar_size, xbar_size)

        # Calculate signs of all weights at once
        signs = np.sign(weights_matrix)

        # Prepare weights with positive magnitudes
        abs_weights = np.abs(weights_matrix)

        # Convert all weights to fixed-point representation
        int_weights = np.vectorize(lambda w: float_to_fixed(w, self.data_config.frac_bits))(abs_weights)

        # Initialize the output array
        xbar_weights = np.zeros((self.mvmu_config.num_rram_xbar_per_mvmu, xbar_size, xbar_size))

        # Process each crossbar
        for k in range(self.mvmu_config.num_rram_xbar_per_mvmu):
            # Extract bits for this crossbar (still need to loop over k)
            xbar_int_weights = np.vectorize(
                lambda w: extract_bits(w, self.mvmu_config.stored_bit[k], self.mvmu_config.stored_bit[k + 1])
            )(int_weights)

            # Convert to conductance values (vectorized)
            conductance_values = np.vectorize(
                lambda w: int_to_conductance(
                    w,
                    self.mvmu_config.bits_per_cell[k],
                    self.mvmu_config.xbar_config.rram_conductance_min,
                    self.mvmu_config.xbar_config.rram_conductance_max,
                )
            )(xbar_int_weights)

            # Apply signs and store in result array
            xbar_weights[k] = signs * conductance_values

        # Load the processed weights into the xbar array
        self.rram_xbar_array.load_weights(xbar_weights)

    def _execute_mvm(self):
        """Execute a detailed matrix-vector multiplication instruction"""

        # Step 1: Reset the output register array
        self.output_register_array.reset()

        # Step 2: Based on data_width and DAC resolution, do Bit slicing
        num_iterations = int(np.ceil(self.config.data_width / self.mvmu_config.dac_config.resolution))
        for i in range(num_iterations):

            # Step 2: Read from the input register array
            sliced_digital_activation = self.input_register_array.read(self.mvmu_config.dac_config.resolution)

            # Step 3: DAC conversion
            dac_output = self.dac_array.convert(sliced_digital_activation)

            # Step 4: RRAM crossbar multiplication
            xbar_output_pos, xbar_output_neg = self.rram_xbar_array.execute_mvm(dac_output)

            # Step 5: Do Sample and Hold (only to count for energy)
            self.snh_array_pos.sample()
            self.snh_array_neg.sample()

            # Step 6: MUX selection
            for j in range(self.mvmu_config.num_columns_per_adc):
                mux_output_pos = self.mux_array_pos.select(xbar_output_pos, j)
                mux_output_neg = self.mux_array_neg.select(xbar_output_neg, j)

                # Step 7: ADC conversion
                adc_output = self.adc_array.convert(mux_output_pos, mux_output_neg)

                # Step 8: Read current value from output register array
                mask = np.arange(j, self.mvmu_config.xbar_config.xbar_size, self.mvmu_config.num_columns_per_adc)
                current_output = self.output_register_array.read(mask)

                # Step 9: SNA operation
                sna_output = self.sna_array.calculate(adc_output, current_output, i)

                # Step 10: Write back to the output register array
                self.output_register_array.write(sna_output, mask)

        # Step 11: Do the clipping since we preserved full precision during the shift and add
        # No code here, just to use read_clipped method in the output register array

    def get_stats(self) -> Stats:
        return self.stats.get_stats(self.xbars + self.adcs + self.dacs)
