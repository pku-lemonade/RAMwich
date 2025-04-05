from typing import List, Dict, Any
from pydantic import BaseModel, Field
from .config import IMAConfig
from .blocks.adc import ADC
from .blocks.dac import DAC
from .xbar import Xbar
from .stats import Stats

class IMAStats(BaseModel):
    op_counts: Dict[str, int] = Field(default_factory=dict, description="Operation counts by type")

    def get_stats(self, ima_id: int) -> Stat:
        stats = Stat()
        stats.latency = 0.0
        stats.energy = 0.0
        stats.area = 0.0
        stats.op_counts = self.op_counts
        return stats

class IMA:
    """
    In-Memory Accelerator containing multiple crossbar arrays with detailed hardware simulation.
    """
    def __init__(self, id: int = 0, num_xbars: int = 4, ima_config=None, adc_config=None, dac_config=None):
        # Basic IMA properties
        self.id = id
        self.ima_config = ima_config if ima_config else IMAConfig()

        # Initialize Xbar arrays
        self.xbars = [Xbar(i, self.ima_config.xbar_size if hasattr(self.ima_config, 'xbar_size') else 32)
                     for i in range(num_xbars)]

        # Statistics tracking
        self.stats = IMAStats()

        # Initialize sub-components
        self.adcs = [ADC(adc_config, i)
                    for i in range(int(self.ima_config.xbar_size // 16 * 2))]
        self.dacs = [DAC(dac_config)
                    for _ in range(self.ima_config.xbar_size)]

        # Memory components
        self.xbar_memory = [[0.0 for _ in range(self.ima_config.xbar_size)]
                           for _ in range(self.ima_config.xbar_size)]
        self.data_memory = [0] * self.ima_config.dataMem_size
        self.instruction_memory = [self._create_dummy_instr()] * self.ima_config.instrnMem_size

        # Pipeline stages
        self.pipeline_stages = ['fet', 'dec', 'ex']
        self.current_stage = 'fet'

        # Performance tracking
        self.compute_cycles = 0
        self.mem_access_cycles = 0
        self.instruction_count = 0
        self.energy_consumption = 0

    def __repr__(self):
        return f"IMA({self.id}, xbars={len(self.xbars)})"

    def execute_mvm(self, xbar_values: List[int]):
        """Execute a Matrix-Vector Multiplication operation on Xbar level"""
        # High-level MVM operation that uses the first xbar
        self.stats.increment_op_count('mvm')

        # Store values in the first xbar
        if self.xbars:
            self.xbars[0].set_values(xbar_values)

        return True

    def execute_instruction(self, instruction):
        """Execute a single instruction in the IMA"""
        cycles = 0
        energy = 0

        # Process each pipeline stage
        for stage in self.pipeline_stages:
            self.current_stage = stage
            stage_cycles, stage_energy = self._process_stage(instruction)
            cycles += stage_cycles
            energy += stage_energy

        # Update stats
        self.instruction_count += 1
        self.compute_cycles += cycles
        self.energy_consumption += energy

        return cycles

    def _process_stage(self, instruction):
        """Process a specific pipeline stage"""
        if self.current_stage == 'fet':
            return 1, 0.1  # Fetch takes 1 cycle and minimal energy

        elif self.current_stage == 'dec':
            return 1, 0.2  # Decode takes 1 cycle and more energy than fetch

        elif self.current_stage == 'ex':
            # Execution depends on instruction type
            opcode = instruction['opcode']

            if opcode in ['ld', 'st']:
                # Memory operation
                self.mem_access_cycles += self.ima_config.dataMem_lat
                return self.ima_config.dataMem_lat, self.ima_config.dataMem_pow_dyn

            elif opcode == 'mvm':
                # Matrix-vector multiplication (most complex operation)
                return self._execute_mvm_instruction(instruction)

            elif opcode == 'alu':
                # ALU operation
                return self.ima_config.alu_lat, self.ima_config.alu_pow_dyn

            else:
                # Default for other operations
                return 1, 0.5

        return 0, 0  # Should never reach here

    def _execute_mvm_instruction(self, instruction):
        """Execute a detailed matrix-vector multiplication instruction"""
        # This involves DACs, crossbar, and ADCs
        cycles = 0
        energy = 0

        # DAC conversion for inputs
        dac_cycles = self.dacs[0].dac_config.lat if self.dacs else 1
        cycles += dac_cycles
        energy += sum(dac.dac_config.pow_dyn for dac in self.dacs)

        # Crossbar computation
        cycles += self.ima_config.xbar_ip_lat
        energy += self.ima_config.xbar_ip_pow

        # ADC conversion for outputs
        adc_cycles = self.adcs[0].adc_config.lat if self.adcs else 1
        cycles += adc_cycles
        energy += sum(adc.adc_config.pow_dyn for adc in self.adcs)

        return cycles, energy

    def update_execution_time(self, execution_time):
        """Update the execution time statistics"""
        self.stats.latency += execution_time

    def get_stats(self) -> Stats:
        stats = Stats()
        for component in self.xbars + self.adcs + self.dacs:
            component_stats = component.get_stats()
            stats.latency += component_stats.latency
            stats.energy += component_stats.energy
            stats.area += component_stats.area
            for op_type, count in component_stats.op_counts.items():
                stats.increment_op_count(op_type, count)
        return stats

    def _create_dummy_instr(self):
        return {"opcode": "nop"}

    def get_total_cycles(self):
        """Return the total execution cycles"""
        return self.compute_cycles + self.mem_access_cycles

    def get_energy_consumption(self):
        """Return the total energy consumption in mJ"""
        # Add energy from sub-components
        adc_energy = sum(adc.get_energy_consumption() for adc in self.adcs)
        dac_energy = sum(dac.get_energy_consumption() for dac in self.dacs)

        return (self.energy_consumption + adc_energy + dac_energy) / 1000  # Convert to mJ
