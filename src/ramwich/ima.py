from .blocks.adc import ADC
from .blocks.dac import DAC
from .blocks.xbar import Xbar
from .config import ADCConfig, DACConfig, IMAConfig
from .stats import Stats


class IMA:
    """
    In-Memory Accelerator containing multiple crossbar arrays with detailed hardware simulation.
    """

    def __init__(
        self, id: int = 0, ima_config: IMAConfig = None, adc_config: ADCConfig = None, dac_config: DACConfig = None
    ):
        # Basic IMA properties
        self.id = id
        self.ima_config = ima_config or IMAConfig()

        # Initialize Xbar arrays
        self.xbars = [
            Xbar(i, self.ima_config.xbar_size if hasattr(self.ima_config, "xbar_size") else 32)
            for i in range(self.ima_config.num_xbars)
        ]

        self.stats = Stats()

        # Initialize sub-components
        self.adcs = [ADC(adc_config, i) for i in range(int(self.ima_config.xbar_size // 16 * 2))]
        self.dacs = [DAC(dac_config) for _ in range(self.ima_config.xbar_size)]

        # Memory components
        self.xbar_memory = [[0.0 for _ in range(self.ima_config.xbar_size)] for _ in range(self.ima_config.xbar_size)]
        self.registers = [0 for _ in range(self.ima_config.num_registers)]

    def __repr__(self):
        return f"IMA({self.id}, xbars={len(self.xbars)})"

    def _execute_mvm(self, instruction):
        """Execute a detailed matrix-vector multiplication instruction"""
        # dispatch DACs, crossbar, and ADCs
        pass

    def get_stats(self) -> Stats:
        return self.stats.get_stats(self.xbars + self.adcs + self.dacs)
