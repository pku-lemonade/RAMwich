import numpy as np

def int2conductance(int_data: int, bits: int, conductance_min: float = 0, conductance_max: float = 1):
    assert int_data < (2 ** bits), "int_data must be less than 2^bits"
    step = (conductance_max - conductance_min) / (2 ** bits - 1)
    return conductance_min + (int_data * step)

# this function is used to convert a bin to its 2s compliment

def twos_complement(binary_string):
    # Ensure the input is a valid binary string
    if not all(bit in '01' for bit in binary_string):
        raise ValueError("Input should be a binary string")

    # Invert the bits
    inverted_bits = ''.join('1' if bit == '0' else '0' for bit in binary_string)

    # Convert to integer, add 1, and convert back to binary
    twos_complement_value = bin(int(inverted_bits, 2) + 1)[2:]

    # Ensure the result has the same length as the input
    return twos_complement_value.zfill(len(binary_string))


def bin2int (binary_string: str, bits: int, compliment: bool = True):
    val = int (binary_string,2)
    if compliment:                         # if the given string is in twos compliment, do the sign computation
        if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << bits)        # compute negative value
    return val

def int2bin (int_data: int, bits: int):
    data_str = bin(int_data & (2 ** bits - 1))[2:].zfill(bits)
    return data_str

def float2fixed (float_data: float, int_bits: int, frac_bits: int):
    temp = float_data * (2 ** frac_bits)
    temp = int(round(temp))
    return int2bin(temp, (int_bits + frac_bits))

def fixed2float (binary_string: str, int_bits: int, frac_bits: int, compliment: bool = True):
    temp = bin2int (binary_string, (int_bits + frac_bits), compliment)
    return float(temp) / (2 ** frac_bits)