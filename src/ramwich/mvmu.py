from typing import Union

import numpy as np
from numpy.typing import NDArray

from .blocks.adc import ADCArray
from .blocks.dac import DACArray
from .blocks.inreg import InputRegisterArray
from .blocks.mux import MuxArray
from .blocks.outreg import OutputRegisterArray
from .blocks.rram_xbar import RRAMXbarArray
from .blocks.sna import SNAArray
from .blocks.snh import SNHArray
from .blocks.sram_cim_unit import SRAMCIMUnitArray
from .config import Config
from .stats import Stats, StatsDict
from .utils.data_convert import extract_bits, float_to_fixed, int_to_conductance


class MVMU:
    """
    Matrix-Vector Multiply unit with multiple crossbar arrays with detailed hardware simulation.
    """

    def __init__(self, id: int, config: Config):
        # Basic MVMU properties
        self.id = id
        self.config = config
        self.data_config = self.config.data_config
        self.mvmu_config = self.config.mvmu_config

        # Initialize basic components
        self.input_register_array = InputRegisterArray(self.mvmu_config)
        self.output_register_array = OutputRegisterArray(self.mvmu_config)
        self.sna_array = SNAArray(self.mvmu_config)

        # Initialize RRAM CIM unit components if using RRAM CIM
        if self.mvmu_config.have_rram_xbar:
            # Initialize DACs.
            # MVMU has multiple DACs based on the xbar_size, 1 DAC per column.
            # The same column of different xbars share the same DAC.
            # Positive and negative crossbars also share the same DAC.
            self.dac_array = DACArray(self.mvmu_config)

            # Initialize RRAM crossbar arrays
            self.rram_xbar_array = RRAMXbarArray(self.mvmu_config)

            # Initialize Sample and Hold (SNH) arrays and MUX arrays
            self.snh_array_pos = SNHArray(self.mvmu_config)
            self.snh_array_neg = SNHArray(self.mvmu_config)
            self.mux_array_pos = MuxArray(self.mvmu_config, type="RRAM")
            self.mux_array_neg = MuxArray(self.mvmu_config, type="RRAM")

            # Initialize ADCs.
            # Each xbar has multiple ADCs based on the xbar_size divided by columns per ADC.
            # Number is multiplied by 2 for positive/negative crossbars for normal adcs, evens for positive and odds for negative.
            self.adc_array = ADCArray(self.mvmu_config)

        # Initialize SRAM CIM unit components if using SRAM CIM
        if self.mvmu_config.have_sram_xbar:
            # Initialize SRAM CIM array
            self.sram_cim_unit_array = SRAMCIMUnitArray(self.mvmu_config)

            # Initialize MUX arrays for SRAM
            self.mux_array_sram = MuxArray(self.mvmu_config, type="SRAM")

        # Initialize stats
        self.stats = Stats()

    def __repr__(self):
        return f"MVMU({self.id})"

    def load_weights(self, weights: NDArray[np.float64]):
        """Load weights into the crossbar arrays"""

        # Validate input length
        xbar_size = self.mvmu_config.xbar_config.xbar_size
        if weights.shape != (xbar_size, xbar_size):
            raise ValueError(f"Expected weights shape ({xbar_size}, {xbar_size}), got {weights.shape}")

        # Calculate signs of all weights at once
        signs = np.sign(weights)

        # Prepare weights with positive magnitudes
        abs_weights = np.abs(weights)

        # Convert all weights to fixed-point representation
        int_weights = np.vectorize(float_to_fixed)(abs_weights, self.data_config.weight_frac_bits)

        # Initialize the output array
        rram_xbar_weights = np.zeros((self.mvmu_config.num_rram_xbar_per_mvmu, xbar_size, xbar_size)).astype(np.float64)
        sram_xbar_weights = np.zeros((self.mvmu_config.num_sram_xbar_per_mvmu, xbar_size, xbar_size)).astype(np.int8)

        rram_idx = 0
        sram_idx = 0

        # Process each crossbar
        for k in range(self.mvmu_config.num_xbar_per_mvmu):
            # Extract bits for this crossbar (still need to loop over k)
            xbar_int_weights = np.vectorize(extract_bits)(
                int_weights, self.mvmu_config.stored_bit[k], self.mvmu_config.stored_bit[k + 1]
            )

            if self.mvmu_config.is_xbar_rram[k]:
                # Convert to conductance values (vectorized)
                conductance_values = np.vectorize(int_to_conductance)(
                    xbar_int_weights,
                    self.mvmu_config.bits_per_cell[k],
                    self.mvmu_config.xbar_config.rram_conductance_min,
                    self.mvmu_config.xbar_config.rram_conductance_max,
                )

                # Apply signs and store in result array
                rram_xbar_weights[rram_idx] = signs * conductance_values
                rram_idx += 1

            else:
                # Directly store the integer weights for SRAM crossbars
                sram_xbar_weights[sram_idx] = signs * xbar_int_weights
                sram_idx += 1

        # Load the processed weights into the xbar array
        if self.mvmu_config.have_rram_xbar:
            self.rram_xbar_array.load_weights(rram_xbar_weights)
        if self.mvmu_config.have_sram_xbar:
            self.sram_cim_unit_array.load_weights(sram_xbar_weights)

    def execute_mvm(self):
        """Execute a detailed matrix-vector multiplication instruction

        Note: We use int to represent the fixed point values, the output will be in int format.
        Therefore, the output will have 2 times the number of franctional bits as the input.
        The output register is keeping the full precision of the output.
        """

        # Step 1: Reset the output register array
        self.output_register_array.clean_cells()

        # Step 2: Based on activation_width and DAC resolution, do Bit slicing
        num_iterations = int(np.ceil(self.data_config.activation_width / self.mvmu_config.dac_config.resolution))
        for i in range(num_iterations):
            # Step 2: Read from the input register array
            sliced_digital_activation = self.input_register_array.read(self.mvmu_config.dac_config.resolution)

            # If using RRAM CIM, do the following steps
            if self.mvmu_config.have_rram_xbar:
                # Step 3: DAC conversion
                dac_output = self.dac_array.convert(sliced_digital_activation)

                # Step 4: RRAM crossbar multiplication
                xbar_output_pos, xbar_output_neg = self.rram_xbar_array.execute_mvm(dac_output)

                # Step 5: Do Sample and Hold (only to count for energy)
                self.snh_array_pos.sample()
                self.snh_array_neg.sample()

            # If using SRAM CIM, do the following steps
            if self.mvmu_config.have_sram_xbar:
                # Parallel with step 3, 4, 5, 6, and 7: SRAM crossbar multiplication
                sram_xbar_output = self.sram_cim_unit_array.execute_mvm(sliced_digital_activation)

            # Step 6: MUX selection
            for j in range(self.mvmu_config.num_columns_per_adc):
                # MUX selection for RRAM
                if self.mvmu_config.have_rram_xbar:
                    # Step 6: MUX selection
                    mux_output_pos = self.mux_array_pos.select(xbar_output_pos, j)
                    mux_output_neg = self.mux_array_neg.select(xbar_output_neg, j)

                    # Step 7: ADC conversion
                    adc_output = self.adc_array.convert(mux_output_pos, mux_output_neg)

                # MUX selection for SRAM
                if self.mvmu_config.have_sram_xbar:
                    mux_output_sram = self.mux_array_sram.select(sram_xbar_output, j)

                # Depending on the type of crossbar, the calculation output will be from different sources
                if not self.mvmu_config.have_sram_xbar:
                    # If all crossbars are RRAM, the calculation output is from the ADC
                    calculation_output = adc_output
                elif not self.mvmu_config.have_rram_xbar:
                    # If all crossbars are SRAM, the calculation output is from the SRAM MUX
                    calculation_output = mux_output_sram
                else:
                    # If both crossbars are present, we need to merge the outputs.
                    # This is done by hardware wiring, so it doesn't cost time and energy.
                    calculation_output = np.zeros(
                        (self.mvmu_config.num_xbar_per_mvmu, self.mvmu_config.num_adc_per_xbar)
                    )
                    calculation_output[self.mvmu_config.rram_to_output_map] = adc_output
                    calculation_output[self.mvmu_config.sram_to_output_map] = mux_output_sram

                # Step 8: Read current value from output register array
                mask = np.arange(j, self.mvmu_config.xbar_config.xbar_size, self.mvmu_config.num_columns_per_adc)
                current_output = self.output_register_array.read(mask)

                # Step 9: SNA operation
                sna_output = self.sna_array.calculate(calculation_output, current_output, i)

                # Step 10: Write back to the output register array
                self.output_register_array.write(sna_output, mask)

    def write_to_inreg(self, start: int, value: Union[NDArray[np.int32], int]):
        """Write values to the input register array"""
        self.input_register_array.write(value, start)

    def read_from_outreg(self, start: int, length: int):
        """Read the clipped output from the output register array

        We kept full precision of the output in the output register array when calculating.
        Therefore when core reads the output register array, it should only read the clipped output.
        Needs to do a right shift to discard fractional bits of LSBs.

        On hardware, the core just reads the middle bits of the output register array. No additional energy cost.
        """
        indices = np.arange(start, start + length)
        return self.output_register_array.read(indices) >> self.data_config.weight_frac_bits

    def reset(self):
        """Reset the MVMU to its initial state"""
        self.rram_xbar_array.reset()
        self.dac_array.reset()
        self.adc_array.reset()
        self.input_register_array.reset()
        self.output_register_array.reset()
        self.snh_array_pos.reset()
        self.snh_array_neg.reset()
        self.mux_array_pos.reset()
        self.mux_array_neg.reset()
        self.sna_array.reset()

    def get_stats(self) -> StatsDict:
        """Get statistics for this MVMU and its components"""
        stats_dict = StatsDict()
        if self.mvmu_config.have_rram_xbar:
            stats_dict.merge(self.rram_xbar_array.get_stats())
            stats_dict.merge(self.dac_array.get_stats())
            stats_dict.merge(self.adc_array.get_stats())
            stats_dict.merge(self.snh_array_pos.get_stats())
            stats_dict.merge(self.snh_array_neg.get_stats())
            stats_dict.merge(self.mux_array_pos.get_stats())
            stats_dict.merge(self.mux_array_neg.get_stats())
        if self.mvmu_config.have_sram_xbar:
            stats_dict.merge(self.sram_cim_unit_array.get_stats())
            stats_dict.merge(self.mux_array_sram.get_stats())
        stats_dict.merge(self.input_register_array.get_stats())
        stats_dict.merge(self.output_register_array.get_stats())
        stats_dict.merge(self.sna_array.get_stats())

        return stats_dict
