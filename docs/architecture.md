# RAMwich Architecture Overview

This document provides a comprehensive overview of the RAMwich simulator architecture, design principles, and implementation details.

## 🏗️ System Architecture

### High-Level Overview

RAMwich simulates a hierarchical SRAM-based Compute-in-Memory (CIM) architecture with the following key components:

```
┌─────────────────────────────────────────────────────────────┐
│                        RAMwich Node                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │    Tile 0   │  │    Tile 1   │  │    Tile N   │          │
│  │             │  │             │  │             │          │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │          │
│  │ │ Core 0  │ │  │ │ Core 0  │ │  │ │ Core 0  │ │          │
│  │ │ ┌─────┐ │ │  │ │ ┌─────┐ │ │  │ │ ┌─────┐ │ │          │
│  │ │ │MVMU │ │ │  │ │ │MVMU │ │ │  │ │ │MVMU │ │ │          │
│  │ │ └─────┘ │ │  │ │ └─────┘ │ │  │ │ └─────┘ │ │          │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Network-on-Chip (NoC)                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 DRAM Controller                     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Component Hierarchy

1. **Node**: Top-level processing unit
2. **Tile**: Collection of cores with shared resources
3. **Core**: Processing unit with multiple MVMUs
4. **MVMU**: Matrix-Vector Multiplication Unit (compute engine)

## 🧠 Core Components

### Matrix-Vector Multiplication Unit (MVMU)

The MVMU is the fundamental compute unit in RAMwich, implementing SRAM-based CIM operations.

#### MVMU Architecture
```
┌─────────────────────────────────────────────────────────┐
│                        MVMU                             │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │  Input Register │  │ Output Register │               │
│  │     Array       │  │     Array       │               │
│  └─────────────────┘  └─────────────────┘               │
│           │                     ▲                       │
│           ▼                     │                       │
│  ┌─────────────────────────────────────────────────┐    │
│  │              SRAM/RRAM Crossbar                 │    │
│  │                                                 │    │
│  │  ┌───┐ ┌───┐ ┌───┐     ┌───┐                    │    │
│  │  │ W │ │ W │ │ W │ ... │ W │  ← Weight Storage  │    │
│  │  └───┘ └───┘ └───┘     └───┘                    │    │
│  │    │     │     │         │                      │    │
│  │    ▼     ▼     ▼         ▼                      │    │
│  │  ┌─────────────────────────────┐                │    │
│  │  │     Analog Computation      │                │    │
│  │  └─────────────────────────────┘                │    │
│  └─────────────────────────────────────────────────┘    │
│                          │                              │
│                          ▼                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │  ADC/DAC + Sample & Hold + Shift & Add          │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

#### Key Features
- **Crossbar Array**: SRAM or RRAM-based weight storage
- **Analog Computation**: In-memory matrix-vector multiplication
- **Mixed-Signal Interface**: ADC/DAC for digital-analog conversion
- **Precision Control**: Configurable bit-width for weights and activations

### Memory Hierarchy

#### SRAM CIM Technology
- **In-Memory Computation**: Weights stored directly in SRAM cells
- **Parallel Operations**: Multiple computations per memory access
- **Energy Efficiency**: Reduced data movement overhead
- **Precision Options**: 1-bit to 8-bit weight storage

#### RRAM Alternative
- **Non-Volatile Storage**: Persistent weight storage
- **High Density**: Smaller cell size compared to SRAM
- **Lower Power**: Reduced leakage current
- **Endurance Considerations**: Limited write cycles

### Network-on-Chip (NoC)

#### Interconnect Architecture
```
Tile 0 ←→ Router ←→ Tile 1
   ↕        ↕        ↕
Router ←→ Router ←→ Router
   ↕        ↕        ↕
Tile 2 ←→ Router ←→ Tile 3
```

#### Features
- **Packet-Based Communication**: Efficient data routing
- **Congestion Management**: Flow control and buffering
- **Energy Modeling**: Accurate power consumption tracking
- **Scalable Topology**: Support for various network configurations

## ⚙️ Configuration System

### Hierarchical Configuration

RAMwich uses a hierarchical configuration system that allows fine-grained control over all system parameters.

#### System-Level Configuration
```yaml
# Top-level architecture
num_tiles_per_node: 4
num_cores_per_tile: 8
num_mvmus_per_core: 2

# Global settings
simulation_mode: "cycle_accurate"
enable_statistics: true
```

#### Data Configuration
```yaml
data_config:
  storage_config: ["SRAM", "RRAM"]  # Per-level memory technology
  weight_format: "Q2.0"            # 2-bit weights, 0 fractional bits
  activation_format: "Q8.0"        # 8-bit activations
  enable_mixed_precision: true
```

#### MVMU Configuration
```yaml
mvmu_config:
  xbar_size: 128                    # Crossbar dimensions
  adc_resolution: 8                 # ADC bit-width
  dac_resolution: 8                 # DAC bit-width
  enable_shift_add: true            # Multi-bit computation
```

### Memory Configuration
```yaml
memory_config:
  sram_config:
    size_per_bank: 1024             # SRAM bank size (KB)
    num_banks: 4                    # Number of banks per core
    access_energy: 1.2e-12          # Energy per access (J)
    leakage_power: 1.0e-9           # Leakage power (W)
  
  dram_config:
    size: 1048576                   # DRAM size (KB)
    bandwidth: 25.6                 # Bandwidth (GB/s)
    access_latency: 100             # Access latency (cycles)
```

## 🔄 Execution Model

### Simulation Flow

1. **Initialization Phase**
   - Parse configuration files
   - Initialize hardware components
   - Load weights and operations

2. **Execution Phase**
   - Process operation queue
   - Execute computations on MVMUs
   - Handle data movement via NoC
   - Update statistics

3. **Analysis Phase**
   - Collect performance metrics
   - Calculate energy consumption
   - Generate visualization data

### Operation Types

#### Compute Operations
- **Matrix-Vector Multiply**: Core CIM operation
- **Element-wise Operations**: Addition, activation functions
- **Pooling Operations**: Max/average pooling for CNNs

#### Data Movement Operations
- **Load**: Transfer data from memory to registers
- **Store**: Write results back to memory
- **Send/Receive**: Inter-tile communication

#### Control Operations
- **Synchronization**: Barrier operations
- **Conditional**: Branch operations
- **Loop**: Iterative computations

## 📊 Performance Modeling

### Energy Models

#### Dynamic Energy
```
E_dynamic = Σ(operations × energy_per_operation)
```

Components:
- **Computation Energy**: MVMU operations
- **Memory Access Energy**: SRAM/DRAM accesses
- **Communication Energy**: NoC data transfers

#### Leakage Energy
```
E_leakage = Σ(component_power × simulation_time)
```

Components:
- **SRAM Leakage**: Static power consumption
- **Logic Leakage**: Control and computation units
- **Interconnect Leakage**: NoC infrastructure

### Timing Models

#### Computation Latency
- **MVMU Operations**: Crossbar access + ADC conversion
- **Pipeline Stages**: Multi-stage computation pipeline
- **Precision Scaling**: Latency scales with bit-width

#### Memory Latency
- **SRAM Access**: Fast, deterministic access
- **DRAM Access**: Higher latency, bandwidth-limited
- **Cache Effects**: Modeling of memory hierarchy

### Area Models

#### Component Areas
- **MVMU Area**: Crossbar + ADC/DAC + control logic
- **Memory Area**: SRAM banks + DRAM interface
- **NoC Area**: Routers + interconnect wires

## 🧪 Accuracy and Validation

### Computational Accuracy

#### Quantization Effects
- **Weight Quantization**: Impact of reduced precision weights
- **Activation Quantization**: Input/output precision effects
- **Accumulation Precision**: Internal computation bit-width

#### Noise Modeling
- **ADC Noise**: Conversion accuracy limitations
- **Process Variation**: Manufacturing tolerance effects
- **Temperature Effects**: Performance variation modeling

### Validation Methodology

#### Reference Models
- **Floating-Point Reference**: High-precision baseline
- **Bit-Accurate Models**: Cycle-accurate simulation
- **Hardware Validation**: Comparison with real implementations

## 🔧 Extensibility

### Plugin Architecture

RAMwich supports extensible components through a plugin system:

#### Custom Memory Technologies
```python
class CustomMemory(MemoryTechnology):
    def __init__(self, config):
        super().__init__(config)
    
    def compute_energy(self, operations):
        # Custom energy model
        pass
    
    def compute_latency(self, operations):
        # Custom timing model
        pass
```

#### Custom Neural Network Layers
```python
class CustomLayer(Layer):
    def __init__(self, config):
        super().__init__(config)
    
    def execute(self, inputs):
        # Custom computation
        pass
```

### Integration APIs

#### Python API
```python
from ramwich import RAMwich, Config

# Programmatic configuration
config = Config()
config.set_architecture(tiles=4, cores=8)
config.set_memory_technology("SRAM")

# Run simulation
simulator = RAMwich(config)
results = simulator.run(inputs)
```

#### C++ Backend
- **Performance-Critical Code**: Implemented in C++
- **Python Bindings**: Seamless integration with Python
- **Memory Management**: Efficient large-scale simulations

## 🎯 Design Principles

### Modularity
- **Component Isolation**: Clear interfaces between components
- **Configurable Parameters**: Extensive customization options
- **Plugin Support**: Easy extension and modification

### Accuracy
- **Cycle-Accurate Simulation**: Precise timing modeling
- **Energy-Accurate Models**: Validated power consumption
- **Bit-Accurate Computation**: Precise quantization effects

### Performance
- **Efficient Implementation**: Optimized simulation algorithms
- **Parallel Execution**: Multi-threaded simulation support
- **Memory Optimization**: Efficient data structures

### Usability
- **Intuitive Configuration**: YAML-based configuration files
- **Rich Visualization**: Comprehensive analysis tools
- **Extensive Documentation**: Complete usage guides