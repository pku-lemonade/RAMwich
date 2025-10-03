# Quick Start Guide

Get up and running with RAMwich in just a few minutes! This guide will walk you through your first simulation.

## 🚀 Your First Simulation

### Step 1: Verify Installation

Make sure RAMwich is properly installed:

```bash
python -c "import ramwich; print('✅ RAMwich ready!')"
```

### Step 2: Run the Example

RAMwich comes with a pre-configured MLP example. Run it with:

```bash
python run.py --config examples/mlp_l4_mnist/config.yaml --ops examples/mlp_l4_mnist/ops.json
```

You should see output similar to:
```
🚀 RAMwich SRAM-CIM Simulator
========================================
🔧 Initializing simulator...
⚡ Running simulation...
============================================================
SIMULATION RESULTS SUMMARY
============================================================
Energy Analysis:
  Total Energy:    8.187e-05 J
  Dynamic Energy:  5.673e-06 J (6.9%)
  Leakage Energy:  7.614e-05 J (93.1%)

Area Analysis:
  Total Area:      35.522 mm²

Performance:
  Simulation Time: 0.123 seconds
============================================================
🎉 Simulation completed successfully!
```

### Step 3: Add Visualization

Generate a visual analysis dashboard:

```bash
python run.py --config examples/mlp_l4_mnist/config.yaml --ops examples/mlp_l4_mnist/ops.json --visualize
```

This creates an HTML dashboard at `output/dashboard.html` that you can open in your browser.

## 🎯 Quick Examples

### Basic Simulation
```bash
# Minimal simulation
python run.py --config config.yaml --ops ops.json
```

### With Pre-trained Weights
```bash
# Include pre-trained model weights
python run.py --config config.yaml --ops ops.json --weights weights.npz
```

### With Input Data
```bash
# Process specific input activation
python run.py --config config.yaml --ops ops.json --weights weights.npz --activation input.npy
```

### Save Results to JSON
```bash
# Export results for further analysis
python run.py --config config.yaml --ops ops.json --json-output results.json
```

### Quiet Mode
```bash
# Minimal output for batch processing
python run.py --config config.yaml --ops ops.json --quiet
```

## 📊 Try the Visualization Demos

RAMwich includes comprehensive visualization examples:

### Quick Visualization Demo
```bash
cd examples
python quick_visualization_demo.py
```

This generates a simple dashboard showing SRAM CIM performance metrics.

### Comprehensive Analysis Demo
```bash
cd examples
python visualization_demo.py
```

This creates a full analysis comparing different configurations and neural network architectures.

## 🧠 Understanding the Output

### Energy Analysis
- **Total Energy**: Combined dynamic and leakage energy consumption
- **Dynamic Energy**: Energy consumed during active operations
- **Leakage Energy**: Static power consumption

### Area Analysis
- **Total Area**: Physical chip area required (mm²)

### Performance Metrics
- **Simulation Time**: Wall-clock time for the simulation
- **Total Cycles**: Number of computation cycles (if available)

## 🔧 Configuration Basics

RAMwich uses YAML configuration files. Here's a minimal example:

```yaml
# config.yaml
num_tiles_per_node: 1
num_cores_per_tile: 1
num_mvmus_per_core: 1

data_config:
  storage_config: ["SRAM", "SRAM"]
  weight_format: "Q2.0"
  activation_format: "Q8.0"
```

Key parameters:
- **Tiles/Cores/MVMUs**: Hardware parallelism levels
- **Storage Config**: Memory technology (SRAM, RRAM, MLC)
- **Data Formats**: Quantization for weights and activations

## 📁 File Formats

### Required Files
- **Config File** (`.yaml`): System configuration
- **Operations File** (`.json`): Computation graph definition

### Optional Files
- **Weights File** (`.npz`): Pre-trained neural network weights
- **Activation File** (`.npy`): Input data for processing

## 🎨 Visualization Features

RAMwich provides rich visualization capabilities:

### Performance Dashboard
- Throughput vs energy efficiency plots
- Latency distribution analysis
- Hardware utilization metrics

### Energy Analysis
- Component-wise energy breakdown
- Dynamic vs leakage power analysis
- Technology comparison charts

### Export Options
- HTML interactive dashboards
- CSV data exports
- JSON result files
- Publication-ready plots

## 🚀 Next Steps

Now that you've run your first simulation, explore more advanced features:

1. **📖 [Usage Guide](usage.md)** - Detailed command-line options
2. **🏗️ [Architecture](architecture.md)** - Understanding RAMwich internals
3. **⚙️ [Configuration](configuration.md)** - Advanced configuration options
4. **🧠 [Neural Networks](neural-networks.md)** - Supported architectures
5. **📊 [Visualization](visualization.md)** - Advanced analysis tools

## 💡 Tips for Success

### Performance Tips
- Use `--quiet` for batch processing
- Save results with `--json-output` for analysis
- Use visualization to understand bottlenecks

### Configuration Tips
- Start with provided examples
- Adjust parallelism based on your workload
- Experiment with different memory technologies

### Analysis Tips
- Compare multiple configurations
- Focus on energy efficiency for mobile applications
- Use area analysis for chip design decisions

## 🆘 Need Help?

- **Documentation**: Browse the [full documentation](README.md)
- **Examples**: Check out more [examples](examples.md)
- **Issues**: Report problems on [GitHub](https://github.com/pku-lemonade/RAMwich/issues)