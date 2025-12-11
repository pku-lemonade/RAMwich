# RAMwich

RAMwich is a cycle-accurate simulator designed for heterogeneous Compute-in-Memory (CiM) architectures, supporting both RRAM and SRAM technologies. It allows researchers and developers to model, simulate, and evaluate the performance, energy efficiency, and area of various CiM configurations. Built with `simpy`, it provides a flexible event-driven simulation environment.

## Features

- **Heterogeneous Architecture Support**: Model architectures with both RRAM and SRAM CiM macros.
- **Cycle-Accurate Simulation**: Precise timing analysis using `simpy`.
- **Configurable**: Define architecture structure and component specifications via JSON and YAML configuration files.
- **Detailed Metrics**: Obtain statistics on latency, energy consumption, and area.
- **Workload Support**: Run neural network workloads defined by operation sequences and parameters.

## Workflow

RAMwich works in conjunction with the [Hybrid-CiM-compiler]. The compiler takes a high-level model description (e.g., in C++) and generates the necessary input files for the simulator:

1.  **Operations (`ops.json`)**: The sequence of operations to be executed by the hardware.
2.  **Weights (`weights.npz`)**: The model parameters (converted from `weights.json`).
3.  **Configuration (`config.json`)**: The architecture configuration derived from the compilation process.

To generate these files, refer to the `Hybrid-CiM-compiler` documentation.

## Architecture Overview

RAMwich models a hierarchical architecture composed of the following levels:

1.  **Node**: The top-level entity containing multiple tiles.
2.  **Tile**: Contains a set of Cores, an eDRAM buffer, and a Network-on-Chip (NoC) router for inter-tile communication.
3.  **Core**: The processing unit containing a set of MVMUs (Matrix-Vector Multiplication Units), a Vector Functional Unit (VFU) for non-linear operations, and local registers.
4.  **MVMU**: The compute engine (Crossbar array) responsible for matrix-vector multiplication.

## Configuration

RAMwich uses a dual-configuration system:

*   **JSON Configuration**: Defines the high-level architecture structure (e.g., number of nodes, tiles, cores, MVMUs).
*   **YAML Configuration**: Defines the detailed hardware specifications for components (e.g., ADC/DAC precision, memory sizes, energy costs).

Key configuration parameters include:
*   `num_nodes`: Number of nodes in the system.
*   `num_tiles_per_node`: Number of tiles per node.
*   `num_cores_per_tile`: Number of cores per tile.
*   `num_mvmus_per_core`: Number of MVMUs per core.

## Supported Operations

The simulator supports a variety of operations defined in JSON format:

*   **Memory Operations**: `load`, `store`, `set`, `copy`
*   **Compute Operations**:
    *   `mvm`: Matrix-Vector Multiplication using the crossbar arrays.
    *   `fvfu`: Floating-point Vector Functional Unit operations (e.g., `add`, `mul`, `sig`, `tanh`, `relu`).
    *   `ivfu`: Integer Vector Functional Unit operations (e.g., `add`, `sub`, `and`, `or`).
*   **Control Flow**: `hlt` (Halt).
*   **Communication**: `send`, `receive` (Inter-tile communication).

## Installation

1.  Clone the repository.
2.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the simulator, use `run.py` with the necessary configuration and data files. Ensure that the `src` directory is in your `PYTHONPATH`.

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python run.py \
  --JSONConfig <path_to_json_config> \
  --YAMLConfig <path_to_yaml_config> \
  --ops <path_to_ops_json> \
  [--params <path_to_params_npz>] \
  [--activation <path_to_activation_npy>] \
  [--timeout <cycles>] \
  [--no-debug]
```

### Arguments

-   `--JSONConfig`: Path to the JSON configuration file defining the architecture structure.
-   `--YAMLConfig`: Path to the YAML configuration file defining component specifications.
-   `--ops`: Path to the JSON file containing the operations to be executed.
-   `--params`: (Optional) Path to the NPZ file containing model parameters (weights).
-   `--activation`: (Optional) Path to the NPY file containing input activations.
-   `--timeout`: (Optional) Simulation timeout in cycles (default: 100000).
-   `--no-debug`: (Optional) Disable debug monitoring.

## Output Metrics

Upon completion, RAMwich provides a summary of statistics, including:

*   **Activation Count**: Total number of activations.
*   **Dynamic Energy**: Energy consumed by active switching and computation.
*   **Leakage Energy**: Static energy consumption over the simulation time.
*   **Area**: Total silicon area estimate.

## Testing

You can run various tests to verify the simulator's functionality.

**Test loading operations and weights:**

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python tests/test_load.py
```

**Test MVMU (Matrix-Vector Multiplication Unit):**

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python tests/test_mvmu.py
```

**Test DRAM Controller:**

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m pytest tests/test_dram_controller.py
```

**Test Core Features:**

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python tests/test_core_features.py
```

**Test Tile Features:**

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python tests/test_tile_features.py
```

**Test MLP on MNIST (Single Batch):**

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python tests/test_mlp_on_mnist_single.py
```

**Test MLP on MNIST (Multi-Batch):**

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python tests/test_mlp_on_mnist_multi.py
```
