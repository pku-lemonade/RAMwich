from typing import List
from .config import IMAConfig
from .adc import AnalogToDigitalConverter
from .dac import DigitalToAnalogConverter

class Xbar:
    """
    Crossbar array in the RAMwich architecture.
    """
    def __init__(self, id: int, size: int = 32):
        self.id = id
        self.size = size
        self.memory = [0] * size

    def __repr__(self):
        return f"Xbar({self.id}, size={self.size})"

    def set_values(self, values: List[int]):
        """Set values in the crossbar"""
        for i, val in enumerate(values):
            if i < self.size:
                self.memory[i] = val
            else:
                break

class IMA:
    """
    In-Memory Accelerator containing multiple crossbar arrays.
    """
    def __init__(self, id: int, num_xbars: int = 4):
        self.id = id
        self.xbars = [Xbar(i) for i in range(num_xbars)]

    def __repr__(self):
        return f"IMA({self.id}, xbars={len(self.xbars)})"

    def execute_mvm(self, xbar_values: List[int]):
        """Execute a Matrix-Vector Multiplication operation"""
        # Placeholder for actual implementation
        # Typically would perform matrix operations using the crossbar arrays

        # For now, just store values in the first xbar
        if self.xbars:
            self.xbars[0].set_values(xbar_values)

        return True

class InMemoryAccelerator:
    """Hardware implementation of the In-Memory Accelerator component"""

    # Supported opcodes
    OP_LIST = ['ld', 'cp', 'st', 'set', 'nop', 'alu', 'alui', 'mvm', 'vvo', 'hlt', 'jmp', 'beq', 'alu_int', 'crs']
    ALU_OP_LIST = ['add', 'sub', 'sna', 'mul', 'sigmoid']

    def __init__(self, ima_config=None, adc_config=None, dac_config=None, ima_id=0):
        self.ima_config = ima_config if ima_config else IMAConfig()
        self.ima_id = ima_id

        # Initialize sub-components
        self.adcs = [AnalogToDigitalConverter(adc_config, i) for i in range(int(self.ima_config.xbar_size // 16 * 2))]
        self.dacs = [DigitalToAnalogConverter(dac_config) for _ in range(self.ima_config.xbar_size)]

        # Memory components
        self.xbar_memory = [[0.0 for _ in range(self.ima_config.xbar_size)] for _ in range(self.ima_config.xbar_size)]
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

    def _create_dummy_instr(self):
        """Create a dummy instruction data structure"""
        return {
            'opcode': self.OP_LIST[0],
            'aluop': self.ALU_OP_LIST[0],
            'd1': 0,
            'r1': 0,
            'r2': 0,
            'r3': 0,
            'vec': 0,
            'imm': 0,
            'xb_nma': 0
        }

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
                return self._execute_mvm(instruction)

            elif opcode == 'alu':
                # ALU operation
                return self.ima_config.alu_lat, self.ima_config.alu_pow_dyn

            else:
                # Default for other operations
                return 1, 0.5

        return 0, 0  # Should never reach here

    def _execute_mvm(self, instruction):
        """Execute a matrix-vector multiplication operation"""
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

    def get_total_cycles(self):
        """Return the total execution cycles"""
        return self.compute_cycles + self.mem_access_cycles

    def get_energy_consumption(self):
        """Return the total energy consumption in mJ"""
        # Add energy from sub-components
        adc_energy = sum(adc.get_energy_consumption() for adc in self.adcs)
        dac_energy = sum(dac.get_energy_consumption() for dac in self.dacs)

        return (self.energy_consumption + adc_energy + dac_energy) / 1000  # Convert to mJ
