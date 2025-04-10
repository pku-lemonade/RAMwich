from .blocks.adc import ADC
from .blocks.dac import DAC
from .blocks.xbar import Xbar
from .config import DataConfig, DACConfig, XBARConfig, ADCConfig, MVMUConfig
from .stats import Stats


class MVMU:
    """
    Matrix-Vector Multiply unit with multiple crossbar arrays with detailed hardware simulation.
    """

    def __init__(
        self, id: int = 0,
        data_config: DataConfig = None,
        mvmu_config: MVMUConfig = None,
        dac_config: DACConfig = None,
        xbar_config: XBARConfig = None,
        adc_config: ADCConfig = None
    ):
        # Basic MVMU properties
        self.id = id
        self.data_config = data_config or DataConfig()
        self.dac_config = dac_config or DACConfig()
        self.xbar_config = xbar_config or XBARConfig()
        self.adc_config = adc_config or ADCConfig()
        self.mvmu_config = mvmu_config or MVMUConfig()

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

    def _execute_mvm(self, instruction):
        """Execute a detailed matrix-vector multiplication instruction"""
        # dispatch DACs, crossbar, and ADCs
        pass

    def get_stats(self) -> Stats:
        return self.stats.get_stats(self.xbars + self.adcs + self.dacs)
