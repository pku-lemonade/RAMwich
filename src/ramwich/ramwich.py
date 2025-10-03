import json
import logging
import os
import re
from typing import Union

import numpy as np
import simpy
import yaml
from numpy.typing import NDArray

from .config import Config
from .node import Node
from .ops import CoreOp, Operation, TileOp
from .stats import StatsDict

# Configure logging
logger = logging.getLogger(__name__)


class RAMwich:
    """
    RAMwich: A comprehensive simulator for SRAM-based Compute-in-Memory (CIM) architectures.
    
    This class provides the main interface for configuring, running, and analyzing
    SRAM CIM simulations with support for various neural network architectures.
    
    Args:
        config_file (str): Path to YAML configuration file
        ops_file (str): Path to JSON operations file
        weights_file (str, optional): Path to NPZ weights file
        quiet (bool, optional): Suppress verbose output. Defaults to False.
        
    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If file formats are unsupported
        ConfigurationError: If configuration is invalid
        
    Example:
        >>> simulator = RAMwich("config.yaml", "ops.json", "weights.npz")
        >>> simulator.run()
        >>> stats = simulator.get_stats()
    """
    
    def __init__(self, config_file: str, ops_file: str, weights_file: str = None, quiet: bool = False):
        # Store quiet flag
        self.quiet = quiet
        
        # Initialize default values
        self.config = None
        self.env = None
        self.nodes = []
        
        try:
            # Validate input parameters
            self._validate_inputs(config_file, ops_file, weights_file)
            
            # Load configuration from file
            self.config = self._load_configuration(config_file)
            
            # Initialize simulation environment
            self.env = simpy.Environment()
            
            # Build the hierarchical architecture
            self.nodes = self._build_architecture()
            
            # Load operations from file
            self.load_operations(ops_file)
            
            # Load weights if provided
            if weights_file:
                self.load_weights(weights_file)
                
            if not self.quiet:
                logger.info(f"RAMwich simulator initialized successfully")
                logger.info(f"Architecture: {len(self.nodes)} nodes, "
                          f"{sum(len(node.tiles) for node in self.nodes)} tiles")
                
        except Exception as e:
            logger.error(f"Failed to initialize RAMwich simulator: {e}")
            raise
    
    def _validate_inputs(self, config_file: str, ops_file: str, weights_file: str = None) -> None:
        """
        Validate input file paths and formats.
        
        Args:
            config_file: Path to configuration file
            ops_file: Path to operations file
            weights_file: Optional path to weights file
            
        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If file formats are unsupported
        """
        # Check required files exist
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        if not os.path.exists(ops_file):
            raise FileNotFoundError(f"Operations file not found: {ops_file}")
            
        if weights_file and not os.path.exists(weights_file):
            raise FileNotFoundError(f"Weights file not found: {weights_file}")
        
        # Validate file formats
        if not config_file.endswith((".yaml", ".yml", ".json")):
            raise ValueError(f"Unsupported config format: {config_file}. Use YAML or JSON.")
            
        if not ops_file.endswith(".json"):
            raise ValueError(f"Unsupported ops format: {ops_file}. Use JSON.")
            
        if weights_file and not weights_file.endswith((".npz", ".npy")):
            raise ValueError(f"Unsupported weights format: {weights_file}. Use NPZ or NPY.")
    
    def _load_configuration(self, config_file: str) -> Config:
        """
        Load and validate configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Config: Validated configuration object
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith((".yaml", ".yml")):
                    config_data = yaml.safe_load(f)
                else:  # JSON
                    config_data = json.load(f)
            
            # Validate configuration using Pydantic
            config = Config.model_validate(config_data)
            
            # Additional validation
            self._validate_configuration(config)
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_file}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_file}: {e}")
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def _validate_configuration(self, config: Config) -> None:
        """
        Perform additional configuration validation.
        
        Args:
            config: Configuration object to validate
            
        Raises:
            ValueError: If configuration values are invalid
        """
        if config.num_nodes <= 0:
            raise ValueError("Number of nodes must be positive")
            
        if config.num_tiles_per_node <= 0:
            raise ValueError("Number of tiles per node must be positive")
            
        if config.num_cores_per_tile <= 0:
            raise ValueError("Number of cores per tile must be positive")
            
        if config.num_mvmus_per_core <= 0:
            raise ValueError("Number of MVMUs per core must be positive")
        
        # Validate memory configurations
        if hasattr(config, 'memory_config'):
            if config.memory_config.sram_size <= 0:
                raise ValueError("SRAM size must be positive")
                
        # Add more validation as needed

    def _build_architecture(self) -> list[Node]:
        """
        Build the hierarchical architecture based on configuration.
        
        Returns:
            list[Node]: List of configured nodes
            
        Raises:
            RuntimeError: If architecture building fails
        """
        try:
            nodes = []
            
            for node_id in range(self.config.num_nodes):
                if not self.quiet:
                    logger.debug(f"Building node {node_id}")
                    
                node = Node(id=node_id, config=self.config)
                nodes.append(node)
            
            if not nodes:
                raise RuntimeError("No nodes were created")
                
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to build architecture: {e}")
            raise RuntimeError(f"Architecture building failed: {e}")

    def get_node(self, node_id: int) -> Node:
        """
        Get a node by ID with bounds checking.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node: The requested node
            
        Raises:
            IndexError: If node_id is out of bounds
        """
        if not 0 <= node_id < len(self.nodes):
            raise IndexError(f"Node ID {node_id} out of bounds. Available nodes: 0-{len(self.nodes)-1}")
            
        return self.nodes[node_id]

    def load_operations(self, file_path: str) -> None:
        """
        Load operations from a JSON file and organize by node/tile/core hierarchy.
        
        Args:
            file_path: Path to JSON operations file
            
        Raises:
            FileNotFoundError: If operations file doesn't exist
            ValueError: If operations file format is invalid
            RuntimeError: If operation loading fails
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Operations file not found: {file_path}")

            with open(file_path, 'r') as f:
                if file_path.endswith(".json"):
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}. Only JSON is supported.")

            if not isinstance(data, list):
                raise ValueError("Operations file must contain a list of operations")

            operations_loaded = 0
            
            # Convert raw data to operation objects and organize by node/tile/core
            for i, op_data in enumerate(data):
                try:
                    # Parse the operation using Pydantic discriminated union
                    operation = Operation.model_validate({"op": op_data})
                    op = operation.op
                    
                    # Route operation to appropriate component
                    self._route_operation(op)
                    operations_loaded += 1
                    
                except ValueError as e:
                    logger.warning(str(e))
                except Exception as e:
                    logger.warning(f"Failed to load operation {i}: {e}")
                    if not self.quiet:
                        logger.debug(f"Problematic operation data: {op_data}")
                    continue

            if operations_loaded == 0:
                raise RuntimeError("No operations were successfully loaded")
                
            if not self.quiet:
                logger.info(f"Loaded {operations_loaded} operations from {file_path}")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in operations file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to load operations from {file_path}: {e}")
            raise
    
    def _validate_operation_targets(self, op) -> None:
        """
        Validate that operation targets exist in the architecture.
        
        Args:
            op: Operation object to validate
            
        Raises:
            ValueError: If operation targets are invalid
        """
        # Check node bounds
        if hasattr(op, 'node') and not 0 <= op.node < len(self.nodes):
            raise ValueError(f"Operation targets invalid node {op.node}")
            
        # Check tile bounds
        if hasattr(op, 'tile'):
            node = self.nodes[op.node] if hasattr(op, 'node') else self.nodes[0]
            if not 0 <= op.tile < len(node.tiles):
                raise ValueError(f"Operation targets invalid tile {op.tile}")
                
        # Check core bounds for core operations
        if hasattr(op, 'core'):
            node = self.nodes[op.node] if hasattr(op, 'node') else self.nodes[0]
            tile = node.tiles[op.tile] if hasattr(op, 'tile') else node.tiles[0]
            if not 0 <= op.core < len(tile.cores):
                raise ValueError(f"Operation targets invalid core {op.core}")
    
    def _route_operation(self, op) -> None:
        """
        Route operation to the appropriate component.
        
        Args:
            op: Operation object to route
        """
        # Access the node and tile
        node = self.get_node(op.node)
        tile = node.get_tile(op.tile)

        # Handle operations by type
        if isinstance(op, TileOp):
            tile.operations.append(op)
        elif isinstance(op, CoreOp):
            core = tile.get_core(op.core)
            core.operations.append(op)
        else:
            logger.warning(f"Unknown operation type: {type(op)}")

    def load_weights(self, file_path: str):
        """Load weights from a NPZ file and organize by node/tile/core/mvmu hierarchy"""
        if not os.path.exists(file_path):
            logger.error(f"Weight file {file_path} not found")
            return

        # Load weights from NPZ file
        if file_path.endswith(".npz"):
            weight_data = np.load(file_path)

            # Define the expected format pattern
            pattern = r"^node(\d+)_tile(\d+)_core(\d+)_mvmu(\d+)$"

            for key in weight_data.files:
                # Validate the key format
                match = re.match(pattern, key)
                if not match:
                    logger.warning(f"Skipping weight with invalid key format: {key}")
                    continue

                # Extract IDs from regex groups
                node_id = int(match.group(1))
                tile_id = int(match.group(2))
                core_id = int(match.group(3))
                mvmu_id = int(match.group(4))

                try:
                    node = self.get_node(node_id)
                    tile = node.get_tile(tile_id)
                    core = tile.get_core(core_id)
                    mvmu = core.get_mvmu(mvmu_id)

                    mvmu.load_weights(weight_data[key])
                except IndexError:
                    logger.error(f"Invalid component ID in key: {key}")
                except Exception as e:
                    logger.error(f"Error loading weights for {key}: {str(e)}")

        else:
            logger.error(f"Unsupported file format: {file_path}. Only NPZ is supported.")

    def load_activation(self, activation: Union[str, NDArray]):
        """Load a activation to input tile"""

        if isinstance(activation, str):
            # If activation is a string, treat it as a file path
            file_path = activation

            if not os.path.exists(file_path):
                logger.error(f"activation file {file_path} not found")
                return

            # Load activation from NPY file
            if not file_path.endswith(".npy"):
                logger.error(f"Unsupported file format: {file_path}. Only NPY is supported.")
                return

            activation_data = np.load(file_path)

            # Validate the activation data
            if activation_data.ndim != 1:
                logger.error(f"Activation data must be a 1D array, got shape {activation_data.shape}")
                return

        elif isinstance(activation, np.ndarray):
            # If activation is a numpy array, use it directly
            activation_data = activation

        else:
            logger.error(f"Unsupported activation type: {type(activation)}. Must be a file path or numpy array.")
            return

        # Validate the length of activation datas
        length = len(activation_data)
        if length > self.config.tile_config.edram_size:
            logger.error(f"Activation data length {length} exceeds EDRAM size {self.config.tile_config.edram_size}")
            return

        # Convert activation data to fixed-point representation (using int)
        activation_data = (activation_data * (1 << self.config.data_config.activation_frac_bits)).astype(np.int32)

        # Load activation data into the first tile of the first node
        node = self.get_node(0)
        tile = node.get_tile(0)
        tile.edram.cells[:length] = activation_data
        tile.dram_controller.valid[:length] = True

    def run(self, activation: Union[str, NDArray] = None):
        """Run the simulation with operations from the specified file"""

        # Load activations if provided
        if activation is not None:
            self.load_activation(activation)
        else:
            # Create a dummy activation if not provided
            dummy_activation = np.zeros(self.config.tile_config.edram_size, dtype=np.int32)
            self.load_activation(dummy_activation)

        # Create and schedule parallel processes for each node
        processes = []
        for node in self.nodes:
            processes.append(self.env.process(node.run(self.env)))

        # Run simulation until all node processes complete
        if processes:
            self.env.run(until=simpy.AllOf(self.env, processes))
        else:
            logger.warning("No node processes to run. Please check the operations file.")

        if not self.quiet:
            self.get_stats().print()

        logger.info(f"Simulation completed at time {self.env.now}")
        # summarize_results(self.nodes)

    def reset(self):
        """Reset the simulator state"""
        self.env = simpy.Environment()
        for node in self.nodes:
            node.reset()

    def get_stats(self) -> StatsDict:
        """Get statistics for this Simulator and its components"""
        stats_dict = StatsDict()
        for node in self.nodes:
            stats_dict.merge(node.get_stats())
        return stats_dict
