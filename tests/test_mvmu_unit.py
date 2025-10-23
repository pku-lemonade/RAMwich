"""
Unit tests for the MVMU block (without loading the full simulator).

This script verifies that weight partitioning runs in parallel in SNA and that the final
result matches a software np.dot under multi-iteration (DAC=1) execution.

How it works:
- Build a tiny MVMU (xbar_size=8) with all-SRAM storage for an 8-bit weight width.
- Partition weights as [0, 4] (two parts of 4 bits each).
- DAC resolution = 1, activation_width = 8 → 8 iterations to exercise SNA.
- Compare hardware result with software np.dot(matrix, activation).

Semantics under test:
- INT: linear dot product: sum_j w[i,j] * a[j]
- EXP: exponential-like shift: sum_j (w[i,j] << a[j])
- MANT: sum of EXP and Linear for that partition

Run:
  PYTHONPATH=RAMwich/src python RAMwich/tests/test_mvmu_unit.py
"""

import numpy as np

from ramwich.config import BitConfig, Config, DACConfig, DataConfig, MVMUConfig, XBARConfig
from ramwich.mvmu import MVMU


def build_config(data_format: list[str]) -> Config:
    """Construct a minimal Config with a single MVMU tailored for unit testing.

    - 8-bit weight width realized as 8 SRAM xbars (one bit per SRAM xbar)
    - Partition: [0, 4] -> two parts of 4 bits each
    - xbar_size: 8 (small and fast for testing)
    - DAC resolution: 1; activation_width: 8 -> eight iterations total
    - Columns per ADC: 8 -> one ADC per xbar
    - Columns per calculator: 8 -> valid SRAM CIM calculator count
    """

    xbar_cfg = XBARConfig(xbar_size=8)
    data_cfg = DataConfig(
        activation_width=4,
        storage_config=[
            BitConfig.SRAM,
            BitConfig.SRAM,
            BitConfig.SRAM,
            BitConfig.SRAM,
            BitConfig.SRAM,
            BitConfig.SRAM,
            BitConfig.SRAM,
            BitConfig.SRAM,
        ],
        weight_partition=[0, 4],
        data_format=data_format,
    )
    mvmu_cfg = MVMUConfig(
        mvmu_type=0,
        data_config=data_cfg,
        xbar_config=xbar_cfg,
        num_columns_per_adc=8,
        num_columns_per_calculator=8,
        dac_config=DACConfig(resolution=1),
    )

    # Provide a top-level Config with only our single mvmu_config
    cfg = Config(mvmu_config=mvmu_cfg, mvmu_configs={0: mvmu_cfg})
    return cfg


def build_mvmu(cfg: Config) -> MVMU:
    return MVMU(id=0, mvmu_type=0, config=cfg)


def run_once(mvmu: MVMU, weights: np.ndarray, activation: np.ndarray) -> np.ndarray:
    # Load weights and activation, run a single MVM
    # MVMU.load_weights expects integer magnitudes for bit extraction
    mvmu.load_weights(weights.astype(np.int32))
    mvmu.write_to_inreg(0, activation.astype(np.int32))
    mvmu.execute_mvm()
    return mvmu.read_from_outreg(0, mvmu.mvmu_config.xbar_config.xbar_size).astype(np.int64)


def make_weight_matrices(xbar_size: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create three weight matrices (low-only, high-only, both) for an 8-bit width with partition [0,4].

    - low-only: values in [0..15]
    - high-only: values in {0,16,32,...,240} (i.e., v<<4 where v in [0..15])
    - both: sum of the two
    All weights are non-negative so the sign handling in load_weights is simple.
    """
    rng = np.random.default_rng(123)
    low = rng.integers(0, 15, size=(xbar_size, xbar_size), dtype=np.int32)
    high = rng.integers(0, 15, size=(xbar_size, xbar_size), dtype=np.int32)
    low_sign = rng.integers(0, 2, size=(xbar_size, xbar_size), dtype=np.int32)
    high_sign = rng.integers(0, 2, size=(xbar_size, xbar_size), dtype=np.int32)
    encoded = (
        low.astype(np.uint32)
        + (high.astype(np.uint32) << 4)
        + (low_sign.astype(np.uint32) << 8)
        + (high_sign.astype(np.uint32) << 9)
    )
    low_sw = np.where(low_sign == 1, -low, low).astype(np.int32)
    high_sw = np.where(high_sign == 1, -high, high).astype(np.int32)
    return low_sw, high_sw, encoded.astype(np.uint32)


def _sw_partition_output(w_part: np.ndarray, activation: np.ndarray, fmt: str, mant_factor: int = 10) -> np.ndarray:
    """Software golden for a single partition according to the requested format.

    - INT:    y = w @ a
    - EXP:    y[i] = sum_j ( w[i,j] << a[j] )
    - MANT:   y = (w @ a) + sum_j ( w[i,j] << a[j] )
    """
    w64 = w_part.astype(np.int64)
    a64 = activation.astype(np.int64)
    y_lin = w64 @ a64
    y_exp = np.sum(w64 << a64[np.newaxis, :], axis=1)
    if fmt == "INT":
        return y_lin
    if fmt == "EXP":
        return y_exp
    if fmt == "MANT":
        return mant_factor * y_lin + y_exp
    raise ValueError(f"Unknown data format: {fmt}")


def test_against_soft_dot(data_format: list[str]):
    cfg = build_config(data_format)
    mvmu = build_mvmu(cfg)

    xbar_size = cfg.mvmu_config.xbar_config.xbar_size
    low_w, high_w, both_w = make_weight_matrices(xbar_size)

    # 8-bit activation, 8 iterations due to DAC resolution=1
    rng = np.random.default_rng(7)
    activation = rng.integers(0, 15, size=(xbar_size,), dtype=np.int32)

    # Hardware result with full (both) weights
    y_hw = run_once(mvmu, both_w, activation)

    # Software golden: per-partition behavior
    fmt_low, fmt_high = data_format
    y_sw_low = _sw_partition_output(low_w, activation, fmt_low)
    y_sw_high = _sw_partition_output(high_w, activation, fmt_high)
    y_sw = (y_sw_low + y_sw_high).astype(np.int64)

    # Always print outputs even if equal
    print("Data format:", data_format)
    print("activation:", activation)
    print("y_hw:", y_hw)
    print("y_sw_low:", y_sw_low)
    print("y_sw_high:", y_sw_high)
    print("y_sw:", y_sw)
    print("diff:", (y_hw - y_sw))

    if not np.array_equal(y_hw, y_sw):
        raise AssertionError("Hardware result does not match software dot product")


def main():
    # Test a few data format configurations
    cases = [["INT", "INT"], ["INT", "EXP"], ["MANT", "EXP"]]

    for df in cases:
        test_against_soft_dot(df)
        print(f"PASS: hardware matches software dot for data_format={df}")


if __name__ == "__main__":
    main()
