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
from .utils.data_convert import extract_bits, int_to_conductance


class MVMU:
    """
    Matrix-Vector Multiply unit with multiple crossbar arrays with detailed hardware simulation.
    """

    def __init__(self, id: int, mvmu_type: int, config: Config):
        # Basic MVMU properties
        self.id = id
        self.mvmu_type = mvmu_type
        self.config = config
        self.mvmu_config = self.config.mvmu_configs.get(
            mvmu_type, config.mvmu_config
        )  # Use default config if not specified
        self.data_config = self.mvmu_config.data_config

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

    def load_weights(self, weights: NDArray[np.uint32]):
        """
        Load weights into the crossbar arrays
        Expected encoding (unsigned integer per cell):
        - Lower bits [0:weight_width): concatenated magnitudes of all partitions according to weight_partition
        - Higher bits [weight_width : weight_width + part_number): one sign bit per partition
            0 => positive, 1 => negative

        Example: 8-bit weights with 2 partitions of 4 bits each => total 10-bit code per cell
        - Bits [0:4]: magnitude for partition 0
        - Bits [4:8]: magnitude for partition 1
        - Bit [8]: sign for partition 0
        - Bit [9]: sign for partition 1
        """

        # Validate input length
        xbar_size = self.mvmu_config.xbar_config.xbar_size
        if weights.shape != (xbar_size, xbar_size):
            raise ValueError(f"Expected weights shape ({xbar_size}, {xbar_size}), got {weights.shape}")

        # Work in int64 for safe shifting and arithmetic
        enc = weights.astype(np.int64)

        # Pull config-derived parameters
        W = self.data_config.weight_width
        wp = self.data_config.weight_partition
        P = len(wp)

        # Extract magnitudes (lower W bits)
        mag_mask = (1 << W) - 1
        magnitudes = enc & mag_mask

        # Prepare per-partition sign factor matrices: +1 if sign bit=0 else -1
        part_sign = []
        for p in range(P):
            sbit = (enc >> (W + p)) & 1
            sgn = 1 - (sbit << 1)  # 1 if 0 else -1
            part_sign.append(sgn.astype(np.int8))

        # Initialize the output array
        rram_xbar_weights = np.zeros((self.mvmu_config.num_rram_xbar_per_mvmu, xbar_size, xbar_size)).astype(np.float64)
        sram_xbar_weights = np.zeros((self.mvmu_config.num_sram_xbar_per_mvmu, xbar_size, xbar_size)).astype(np.int8)

        rram_idx = 0
        sram_idx = 0

        # Helper to map an xbar start bit to its partition index
        def part_index_for_bit(bitpos: int) -> int:
            idx = 0
            for i in range(P):
                if wp[i] <= bitpos:
                    idx = i
                else:
                    break
            return idx

        # Process each crossbar
        for k in range(self.mvmu_config.num_xbar_per_mvmu):
            start = self.mvmu_config.stored_bit[k]
            end = self.mvmu_config.stored_bit[k + 1]
            # Extract magnitude bits for this xbar slice
            xbar_mag = np.vectorize(extract_bits)(magnitudes, start, end)

            # Determine partition sign to apply for this slice
            p_idx = part_index_for_bit(start)
            sgn = part_sign[p_idx].astype(np.int8)

            if self.mvmu_config.is_xbar_rram[k]:
                # Convert to conductance values (vectorized)
                conductance_values = np.vectorize(int_to_conductance)(
                    xbar_mag,
                    self.mvmu_config.bits_per_cell[k],
                    self.mvmu_config.xbar_config.rram_conductance_min,
                    self.mvmu_config.xbar_config.rram_conductance_max,
                )

                # Apply partition sign and store in result array
                rram_xbar_weights[rram_idx] = sgn * conductance_values
                rram_idx += 1

            else:
                # SRAM stores 1-bit per cell; place +1/-1 when the bit is set, else 0
                bit_set = (xbar_mag > 0).astype(np.int8)
                sram_xbar_weights[sram_idx] = (sgn * bit_set).astype(np.int8)
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
        for i in range(self.mvmu_config.num_iterations):
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
                sram_mvm_output, sram_exp_output = self.sram_cim_unit_array.execute(sliced_digital_activation, i)

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
                    mux_mvm_sram = self.mux_array_sram.select(sram_mvm_output, j)
                    mux_exp_sram = self.mux_array_sram.select(sram_exp_output, j)

                # Depending on the type of crossbar, prepare MVM and EXP outputs separately
                if not self.mvmu_config.have_sram_xbar:
                    # If all crossbars are RRAM, the calculation output is from the ADC (all MVM, no EXP)
                    calculation_mvm = adc_output
                    calculation_exp = np.zeros_like(adc_output)
                elif not self.mvmu_config.have_rram_xbar:
                    # If all crossbars are SRAM, separate MVM and EXP from SRAM MUX
                    calculation_mvm = mux_mvm_sram
                    calculation_exp = mux_exp_sram
                else:
                    # If both crossbars are present, we need to merge the outputs.
                    # This is done by hardware wiring, so it doesn't cost time and energy.
                    calculation_mvm = np.zeros((self.mvmu_config.num_xbar_per_mvmu, self.mvmu_config.num_adc_per_xbar))
                    calculation_exp = np.zeros((self.mvmu_config.num_xbar_per_mvmu, self.mvmu_config.num_adc_per_xbar))
                    calculation_mvm[self.mvmu_config.rram_to_output_map] = adc_output
                    calculation_mvm[self.mvmu_config.sram_to_output_map] = mux_mvm_sram
                    calculation_exp[self.mvmu_config.sram_to_output_map] = mux_exp_sram

                # Step 8: Read current value from output register array
                mask = np.arange(j, self.mvmu_config.xbar_config.xbar_size, self.mvmu_config.num_columns_per_adc)
                current_output = self.output_register_array.read(mask)

                # Step 9: SNA operation with separate MVM and EXP inputs
                sna_output = self.sna_array.calculate(calculation_mvm, calculation_exp, current_output, i)

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
        return self.output_register_array.read(indices)

    def read_weights(self) -> NDArray[np.int64]:
        """Read back the weights stored in the crossbar arrays (for testing/debugging)"""
        xbar_size = self.mvmu_config.xbar_config.xbar_size
        data_cfg = self.data_config

        if data_cfg.part_number is None:
            raise RuntimeError("Data configuration is missing partition metadata; weights cannot be reconstructed.")

        # Allocate result tensor (partition, row, column)
        weights = np.zeros((data_cfg.part_number, xbar_size, xbar_size), dtype=np.int64)

        # Helper to map a bit position back to the partition index used during loading
        def part_index_for_bit(bitpos: int) -> int:
            idx = 0
            for p in range(data_cfg.part_number):
                if data_cfg.weight_partition[p] <= bitpos:
                    idx = p
                else:
                    break
            return idx

        rram_idx = 0
        sram_idx = 0
        g_min = self.mvmu_config.xbar_config.rram_conductance_min
        g_max = self.mvmu_config.xbar_config.rram_conductance_max

        for k in range(self.mvmu_config.num_xbar_per_mvmu):
            start = self.mvmu_config.stored_bit[k]
            end = self.mvmu_config.stored_bit[k + 1]
            bits = end - start
            if bits <= 0:
                continue

            part_idx = part_index_for_bit(start)
            shift = start - data_cfg.weight_partition[part_idx]
            if shift < 0:
                raise ValueError("Stored bit configuration is inconsistent with weight partitions.")

            if self.mvmu_config.is_xbar_rram[k]:
                # Recover signed conductance and map back to integer magnitude
                pos = self.rram_xbar_array.pos_xbar[rram_idx]
                neg = self.rram_xbar_array.neg_xbar[rram_idx]
                signed_conductance = pos - neg
                abs_conductance = np.abs(signed_conductance)

                denom = (1 << bits) - 1
                if denom <= 0 or g_max == g_min:
                    magnitude = np.zeros_like(abs_conductance, dtype=np.int64)
                else:
                    step = (g_max - g_min) / denom
                    # Guard against floating error before rounding
                    raw = np.maximum(abs_conductance - g_min, 0.0) / step
                    magnitude = np.rint(raw).astype(np.int64)
                    magnitude = np.clip(magnitude, 0, denom)

                sign = np.zeros_like(signed_conductance, dtype=np.int64)
                sign[signed_conductance > 0] = 1
                sign[signed_conductance < 0] = -1
                sign = np.where(magnitude == 0, 0, sign)
                contrib = (magnitude * sign).astype(np.int64)
                rram_idx += 1
            else:
                # SRAM weights are ternary {-1,0,1}
                pos = self.sram_cim_unit_array.pos_xbar[sram_idx].astype(np.int64)
                neg = self.sram_cim_unit_array.neg_xbar[sram_idx].astype(np.int64)
                contrib = pos - neg
                sram_idx += 1

            # Align the extracted bits back to the partition-local position
            if shift != 0:
                contrib = np.left_shift(contrib, shift)

            weights[part_idx] += contrib

        return weights

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
