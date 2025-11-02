import copy
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
    def __init__(self, json_config_file: str, yaml_config_file: str, ops_file: str, params_file: str = None):
        # Load base configuration from JSON file
        if not os.path.exists(json_config_file):
            raise FileNotFoundError(f"Configuration file {json_config_file} not found")

        with open(json_config_file) as f:
            if json_config_file.endswith(".json"):
                json_config = json.load(f)
                print("config.json loaded")
            else:
                raise ValueError(f"Unsupported config format: {json_config_file}. Use JSON.")

        # Load additional configuration from YAML file
        if not os.path.exists(yaml_config_file):
            raise FileNotFoundError(f"YAML configuration file {yaml_config_file} not found")

        with open(yaml_config_file) as f:
            if yaml_config_file.endswith(".yaml"):
                yaml_config = yaml.safe_load(f)
                print("config.yaml loaded")
            else:
                raise ValueError(f"Unsupported config format: {yaml_config_file}. Use YAML.")

        # Merge configurations (JSON values override YAML values)
        merged_config = self._merge_configs(json_config, yaml_config)

        # Validate the merged configuration
        self.config = Config.model_validate(merged_config)

        self.env = simpy.Environment()

        # Build the hierarchical architecture
        self.nodes: list[Node] = self._build_architecture()

        # Load operations from file
        self.load_operations(ops_file)

        # Load weights if provided
        if params_file:
            self.load_params(params_file)

    def _merge_configs(self, json_config: dict, yaml_config: dict) -> dict:
        """
        Merge YAML and JSON configurations.
        JSON config values override YAML config values for the same keys.
        For nested dictionaries, performs a deep merge.
        """
        base = copy.deepcopy(yaml_config) if yaml_config else {}

        # Deep merge function for nested dictionaries
        def deep_merge(original: dict, override: dict):
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(original.get(key), dict):
                    deep_merge(original[key], value)
                else:
                    original[key] = value

        if json_config:
            deep_merge(base, json_config)

        return base

    def _build_architecture(self) -> list[Node]:
        """Build the hierarchical architecture based on configuration"""
        nodes = []

        for node_id in range(self.config.num_nodes):
            node = Node(id=node_id, config=self.config)
            nodes.append(node)

        return nodes

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def load_operations(self, file_path: str):
        """Load operations from a JSON file and organize by node/tile/core hierarchy"""
        if not os.path.exists(file_path):
            logger.error(f"Operation file {file_path} not found")
            return

        with open(file_path) as f:
            if file_path.endswith(".json"):
                data = json.load(f)
            else:
                logger.error(f"Unsupported file format: {file_path}. Only JSON is supported.")
                return

        # Convert raw data to operation objects and organize by node/tile/core
        for op_data in data:
            try:
                # Parse the operation using Pydantic discriminated union
                operation = Operation.model_validate({"op": op_data})
                op = operation.op

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

            except ValueError as e:
                logger.warning(str(e))

    def load_params(self, file_path: str):
        """Load weights from a NPZ file and organize by node/tile/core/mvmu hierarchy"""
        if not os.path.exists(file_path):
            logger.error(f"Weight file {file_path} not found")
            return

        # Load weights from NPZ file
        if file_path.endswith(".npz"):
            weight_data = np.load(file_path)

            # Define the expected format patterns
            weight_pattern = r"^weight_node(\d+)_tile(\d+)_core(\d+)_mvmu(\d+)$"
            vector_pattern = r"^vector_node(\d+)_tile(\d+)_core(\d+)_reg(\d+)$"
            legacy_weight_pattern = r"^node(\d+)_tile(\d+)_core(\d+)_mvmu(\d+)$"

            for key in weight_data.files:
                weight_match = re.match(weight_pattern, key)
                vector_match = re.match(vector_pattern, key)
                legacy_weight_match = re.match(legacy_weight_pattern, key)

                try:
                    if weight_match or legacy_weight_match:
                        match = weight_match or legacy_weight_match
                        node_id = int(match.group(1))
                        tile_id = int(match.group(2))
                        core_id = int(match.group(3))
                        mvmu_id = int(match.group(4))

                        node = self.get_node(node_id)
                        tile = node.get_tile(tile_id)
                        core = tile.get_core(core_id)
                        mvmu = core.get_mvmu(mvmu_id)
                        mvmu.load_weights(weight_data[key])
                    elif vector_match:
                        node_id = int(vector_match.group(1))
                        tile_id = int(vector_match.group(2))
                        core_id = int(vector_match.group(3))
                        register_id = int(vector_match.group(4))

                        node = self.get_node(node_id)
                        tile = node.get_tile(tile_id)
                        core = tile.get_core(core_id)
                        core.load_vector(register_id, weight_data[key])
                    else:
                        logger.warning(f"Skipping weight with invalid key format: {key}")
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

        # Load activation data into the first tile of the first node
        node = self.get_node(0)
        tile = node.get_tile(0)
        tile.edram.cells[:length] = activation_data
        tile.edram.type_bits[:length] = True  # Assuming activations are floats
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
