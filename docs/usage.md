# Usage Guide

This guide covers the complete command-line interface and usage patterns for RAMwich.

## 🖥️ Command Line Interface

### Basic Syntax

```bash
python run.py --config CONFIG_FILE --ops OPS_FILE [OPTIONS]
```

### Required Arguments

| Argument | Description | Format |
|----------|-------------|---------|
| `--config` | System configuration file | YAML |
| `--ops` | Operations definition file | JSON |

### Optional Arguments

#### Input Files
| Argument | Description | Format | Example |
|----------|-------------|---------|---------|
| `--weights` | Pre-trained model weights | NPZ | `weights.npz` |
| `--activation` | Input activation data | NPY | `input.npy` |

#### Output Options
| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--output-dir` | Output directory | `output` | `results/` |
| `--json-output` | Save results to JSON | None | `results.json` |
| `--visualize` | Generate dashboard | False | (flag) |

#### Execution Options
| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--quiet` | Suppress verbose output | False | (flag) |
| `--log-level` | Logging verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--validate-only` | Only validate files | False | (flag) |

## 📝 Usage Examples

### Basic Simulations

#### Minimal Simulation
```bash
python run.py --config config.yaml --ops ops.json
```

#### With Weights and Input
```bash
python run.py --config config.yaml --ops ops.json \
              --weights model.npz --activation input.npy
```

#### Validate Configuration Only
```bash
python run.py --config config.yaml --ops ops.json --validate-only
```

### Output and Analysis

#### Generate Visualization Dashboard
```bash
python run.py --config config.yaml --ops ops.json --visualize
```

#### Save Results to JSON
```bash
python run.py --config config.yaml --ops ops.json --json-output results.json
```

#### Custom Output Directory
```bash
python run.py --config config.yaml --ops ops.json \
              --visualize --output-dir my_results/
```

### Batch Processing

#### Quiet Mode for Scripts
```bash
python run.py --config config.yaml --ops ops.json --quiet --json-output batch_results.json
```

#### Debug Mode
```bash
python run.py --config config.yaml --ops ops.json --log-level DEBUG
```

### Complete Example
```bash
python run.py --config examples/mlp_l4_mnist/config.yaml \
              --ops examples/mlp_l4_mnist/ops.json \
              --weights examples/mlp_l4_mnist/weights.npz \
              --activation examples/mlp_l4_mnist/activation.npy \
              --visualize --output-dir results/ \
              --json-output results/simulation.json
```

## 📁 File Formats

### Configuration File (YAML)

The configuration file defines the hardware architecture and simulation parameters.

```yaml
# Basic hardware configuration
num_tiles_per_node: 2
num_cores_per_tile: 4
num_mvmus_per_core: 2

# Memory and computation configuration
data_config:
  storage_config: ["SRAM", "SRAM"]  # Memory technology per level
  weight_format: "Q2.0"             # Weight quantization
  activation_format: "Q8.0"         # Activation quantization

# Optional: Advanced configurations
mvmu_config:
  xbar_size: 128
  adc_resolution: 8
  dac_resolution: 8

memory_config:
  sram_size: 1024
  dram_size: 1048576
```

### Operations File (JSON)

The operations file defines the computation graph and data flow.

```json
{
  "operations": [
    {
      "type": "load",
      "node": 0,
      "tile": 0,
      "source": "weights",
      "destination": "mvmu_0"
    },
    {
      "type": "compute",
      "node": 0,
      "tile": 0,
      "operation": "matrix_vector_multiply",
      "inputs": ["input_data"],
      "outputs": ["intermediate_result"]
    }
  ]
}
```

### Weights File (NPZ)

NumPy compressed archive containing pre-trained weights:

```python
import numpy as np

# Create weights
weights = {
    'layer1': np.random.randn(128, 784),
    'layer2': np.random.randn(64, 128),
    'layer3': np.random.randn(10, 64)
}

# Save to NPZ
np.savez('weights.npz', **weights)
```

### Activation File (NPY)

NumPy array containing input data:

```python
import numpy as np

# Create input data (e.g., MNIST image)
activation = np.random.rand(784)  # 28x28 flattened

# Save to NPY
np.save('activation.npy', activation)
```

## 📊 Output Formats

### Console Output

#### Standard Output
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

#### Quiet Mode Output
```
✅ Results saved to: results.json
```

### JSON Output

```json
{
  "total_dynamic_energy": 5.672660218989583e-06,
  "total_leakage_energy": 7.61427588653072e-05,
  "total_area": 35.52202302670939,
  "simulation_time": 0.123,
  "component_stats": {
    "SRAM": {
      "activation_count": 72408,
      "dynamic_energy": 72408.0,
      "leakage_energy": 947165.0,
      "area": 0.09408
    }
  }
}
```

### Visualization Dashboard

When using `--visualize`, RAMwich generates an HTML dashboard with:

- **Performance Analysis**: Throughput, latency, efficiency metrics
- **Energy Breakdown**: Component-wise energy consumption
- **Area Analysis**: Hardware resource utilization
- **Interactive Plots**: Zoom, pan, and explore data
- **Export Options**: Save plots and data

## 🔧 Advanced Usage Patterns

### Batch Processing Script

```bash
#!/bin/bash
# batch_simulate.sh

configs=("config1.yaml" "config2.yaml" "config3.yaml")
ops="ops.json"
output_base="batch_results"

for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    output_dir="${output_base}/run_$i"
    
    echo "Running simulation $i with $config"
    python run.py --config "$config" --ops "$ops" \
                  --quiet --output-dir "$output_dir" \
                  --json-output "$output_dir/results.json"
done

echo "Batch processing complete!"
```

### Configuration Sweep

```python
# config_sweep.py
import yaml
import subprocess
import itertools

# Parameter ranges
tiles = [1, 2, 4]
cores = [1, 2, 4]
storage_configs = [["SRAM", "SRAM"], ["RRAM", "RRAM"], ["SRAM", "RRAM"]]

base_config = {
    'num_tiles_per_node': 1,
    'num_cores_per_tile': 1,
    'num_mvmus_per_core': 1,
    'data_config': {
        'storage_config': ["SRAM", "SRAM"],
        'weight_format': "Q2.0",
        'activation_format': "Q8.0"
    }
}

for i, (t, c, s) in enumerate(itertools.product(tiles, cores, storage_configs)):
    config = base_config.copy()
    config['num_tiles_per_node'] = t
    config['num_cores_per_tile'] = c
    config['data_config']['storage_config'] = s
    
    config_file = f"sweep_config_{i}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Run simulation
    subprocess.run([
        'python', 'run.py',
        '--config', config_file,
        '--ops', 'ops.json',
        '--quiet',
        '--json-output', f'sweep_results_{i}.json'
    ])
```

### Integration with Analysis Tools

```python
# analyze_results.py
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = []
for i in range(10):  # Assuming 10 simulation runs
    with open(f'results_{i}.json', 'r') as f:
        data = json.load(f)
        data['run_id'] = i
        results.append(data)

# Create DataFrame
df = pd.DataFrame(results)

# Analysis
plt.figure(figsize=(10, 6))
plt.scatter(df['total_energy'], df['total_area'])
plt.xlabel('Total Energy (J)')
plt.ylabel('Total Area (mm²)')
plt.title('Energy vs Area Trade-off')
plt.savefig('energy_area_tradeoff.png')
```

## 🚨 Error Handling

### Common Errors and Solutions

#### File Not Found
```
❌ File not found: config.yaml
```
**Solution**: Check file paths and ensure files exist.

#### Invalid Configuration
```
❌ Simulation failed: Invalid configuration parameter
```
**Solution**: Validate YAML syntax and parameter values.

#### Memory Issues
```
❌ Simulation failed: Out of memory
```
**Solution**: Reduce problem size or use a machine with more RAM.

### Validation Mode

Use `--validate-only` to check files without running simulation:

```bash
python run.py --config config.yaml --ops ops.json --validate-only
```

## 🎯 Best Practices

### Performance Optimization
1. Use `--quiet` for batch processing
2. Save intermediate results with `--json-output`
3. Use appropriate logging levels
4. Validate configurations before long runs

### File Organization
```
project/
├── configs/
│   ├── sram_config.yaml
│   └── rram_config.yaml
├── operations/
│   └── mlp_ops.json
├── data/
│   ├── weights.npz
│   └── test_inputs.npy
└── results/
    ├── run1/
    └── run2/
```

### Reproducibility
- Use version control for configurations
- Document parameter choices
- Save complete results with metadata
- Use fixed random seeds when applicable

## 🆘 Getting Help

- **Command Help**: `python run.py --help`
- **Documentation**: [Full documentation](README.md)
- **Examples**: [Example gallery](examples.md)
- **Issues**: [GitHub Issues](https://github.com/pku-lemonade/RAMwich/issues)