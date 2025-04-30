import json
import logging
import os
import re
from typing import List, Union

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
    def __init__(self, config_file: str):
        # Load configuration from file
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} not found")

        with open(config_file, "r") as f:
            if config_file.endswith((".yaml", ".yml")):
                self.config = Config.model_validate(yaml.safe_load(f))
            else:
                raise ValueError(f"Unsupported config format: {config_file}. Use JSON or YAML.")

        self.env = simpy.Environment()

        # Build the hierarchical architecture
        self.nodes: List[Node] = self._build_architecture()

    def _build_architecture(self) -> List[Node]:
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

        with open(file_path, "r") as f:
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
        activation_data = (activation_data * (1 << self.config.data_config.frac_bits)).astype(np.int32)

        # Load activation data into the first tile of the first node
        node = self.get_node(0)
        tile = node.get_tile(0)
        tile.edram.cells[:length] = activation_data
        tile.dram_controller.valid[:length] = True

    def run(self, ops_file: str, weights_file: str = None, activation: Union[str, NDArray] = None):
        """Run the simulation with operations from the specified file"""
        # Load operations into node/tile/core hierarchy

        start_time = self.env.now

        self.load_operations(ops_file)

        # Load weights if provided
        if weights_file:
            self.load_weights(weights_file)

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

        active_cycles = self.env.now - start_time

        stats_dict = self.get_stats()

        stats_dict.print()

        logger.info(f"Simulation completed at time {self.env.now}")
        # summarize_results(self.nodes)

    def get_stats(self) -> StatsDict:
        """Get statistics for this Simulator and its components"""
        stats_dict = StatsDict()
        for node in self.nodes:
            stats_dict.merge(node.get_stats())
        return stats_dict
