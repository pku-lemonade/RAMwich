def int_to_conductance(int_data: int, bits: int, conductance_min: float = 0, conductance_max: float = 1):
    assert int_data < (2**bits), "int_data must be less than 2^bits"
    step = (conductance_max - conductance_min) / (2**bits - 1)
    return conductance_min + (int_data * step)


def float_to_fixed(float_val: float, fractional_bits: int) -> int:
    assert fractional_bits >= 0, "fractional_bits must be non-negative"
    scaling_factor = 1 << fractional_bits
    scaled_value = float_val * scaling_factor
    fixed_point_val = round(scaled_value)
    return fixed_point_val


def fixed_to_float(fixed_point_val: int, fractional_bits: int) -> float:
    assert fractional_bits >= 0, "fractional_bits must be non-negative"
    scaling_factor = 1 << fractional_bits
    float_val = fixed_point_val / scaling_factor
    return float_val


def extract_bits(val: int, start_index: int, end_index: int) -> int:
    assert start_index >= 0, "start_index must be non-negative"
    assert start_index < end_index, "start_index must be strictly less than end_index"
    num_bits_to_extract = end_index - start_index
    mask = (1 << num_bits_to_extract) - 1
    shifted_value = val >> start_index
    extracted_value = shifted_value & mask
    return extracted_value
