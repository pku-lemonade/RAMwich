# Installation Guide

This guide will help you install RAMwich and its dependencies on your system.

## 📋 Requirements

### System Requirements
- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 1GB free space

### Python Dependencies
RAMwich requires the following Python packages:
- `numpy` >= 1.20.0
- `pyyaml` >= 5.4.0
- `matplotlib` >= 3.3.0 (for visualization)
- `seaborn` >= 0.11.0 (for visualization)
- `pandas` >= 1.3.0 (for data analysis)

## 🚀 Installation Methods

### Method 1: Install from Source (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pku-lemonade/RAMwich.git
   cd ramwich
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv ramwich-env
   source ramwich-env/bin/activate  # On Windows: ramwich-env\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install RAMwich in development mode**:
   ```bash
   pip install -e .
   ```

### Method 2: Install from PyPI (Coming Soon)

```bash
pip install ramwich
```

*Note: PyPI package is not yet available. Use Method 1 for now.*

## 🔧 Verification

### Test Installation
Run the following command to verify your installation:

```bash
python -c "import ramwich; print('RAMwich installed successfully!')"
```

### Run Example Simulation
Test with the provided example:

```bash
python run.py --config examples/mlp_l4_mnist/config.yaml --ops examples/mlp_l4_mnist/ops.json
```

### Run Test Suite
Verify all components work correctly:

```bash
python -m pytest tests/ -v
```

Expected output: All tests should pass (35/35 tests passing).

## 🎨 Optional Dependencies

### Visualization Features
For full visualization capabilities, install additional packages:

```bash
pip install matplotlib seaborn plotly jupyter
```

### Development Tools
For development and testing:

```bash
pip install pytest pytest-cov black flake8 mypy
```

## 🐳 Docker Installation (Alternative)

If you prefer using Docker:

1. **Build the Docker image**:
   ```bash
   docker build -t ramwich .
   ```

2. **Run a simulation**:
   ```bash
   docker run -v $(pwd)/examples:/app/examples ramwich \
     python run.py --config examples/mlp_l4_mnist/config.yaml --ops examples/mlp_l4_mnist/ops.json
   ```

## 🔍 Troubleshooting

### Common Issues

#### ImportError: No module named 'ramwich'
**Solution**: Make sure you installed RAMwich with `pip install -e .` from the project root.

#### ModuleNotFoundError: No module named 'matplotlib'
**Solution**: Install visualization dependencies:
```bash
pip install matplotlib seaborn
```

#### Permission denied errors
**Solution**: Use a virtual environment or install with `--user` flag:
```bash
pip install --user -r requirements.txt
```

#### Python version compatibility
**Solution**: Ensure you're using Python 3.9 or higher:
```bash
python --version
```

### Platform-Specific Notes

#### macOS
- Install Xcode command line tools: `xcode-select --install`
- Consider using Homebrew for Python: `brew install python`

#### Windows
- Use Windows Subsystem for Linux (WSL) for best compatibility
- Install Microsoft Visual C++ Build Tools if needed

#### Linux
- Install development packages: `sudo apt-get install python3-dev build-essential`

## 📊 Performance Optimization

### For Large Simulations
- Install NumPy with optimized BLAS: `pip install numpy[mkl]`
- Use multiple CPU cores: Set `OMP_NUM_THREADS` environment variable
- Consider using a high-memory system for large neural networks

### For Visualization
- Install GPU-accelerated matplotlib backend if available
- Use `--quiet` flag for batch processing to reduce output overhead

## ✅ Next Steps

After successful installation:

1. 📖 Read the [Quick Start Guide](quickstart.md)
2. 🏃 Try the [Usage Examples](usage.md)
3. 🧠 Explore [Neural Network Support](neural-networks.md)
4. 📊 Learn about [Visualization Features](visualization.md)

## 🆘 Getting Help

If you encounter issues during installation:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search [GitHub Issues](https://github.com/pku-lemonade/RAMwich/issues)
3. Create a new issue with your system details and error messages