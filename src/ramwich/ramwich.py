import json
import logging
import os
import re
from typing import List

import numpy as np
import simpy
import yaml

from .config import Config
from .core import Core
from .mvmu import MVMU
from .node import Node
from .ops import CoreOp, Operation, TileOp, Weight
from .stats import Stats
from .tile import Tile
from .utils.visualize import summarize_results

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
        self.stats = Stats()  # Add stats attribute

        # Build the hierarchical architecture
        self.nodes: List[Node] = self._build_architecture()

    def _build_architecture(self) -> List[Node]:
        """Build the hierarchical architecture based on configuration"""
        nodes = []

        for node_id in range(self.config.num_nodes):
            tiles = []

            for tile_id in range(self.config.num_tiles_per_node):
                cores = []

                for core_id in range(self.config.num_cores_per_tile):
                    mvmus = []

                    for mvmu_id in range(self.config.num_mvmus_per_core):
                        # Create MVMU with its xbars
                        mvmu = MVMU(id=mvmu_id, config=self.config)
                        mvmus.append(mvmu)

                    # Create Core with its MVMUs and config
                    core = Core(id=core_id, mvmus=mvmus, config=self.config)
                    cores.append(core)

                # Create Tile with its Cores and config
                tile = Tile(id=tile_id, cores=cores, config=self.config)
                tiles.append(tile)

            # Create Node with its Tiles and config
            node = Node(id=node_id, tiles=tiles, config=self.config)
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

        # Load weights from JSON file
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

    def run(self, ops_file: str, weights_file: str = None):
        """Run the simulation with operations from the specified file"""
        # Load operations into node/tile/core hierarchy
        self.load_operations(ops_file)

        # Create and schedule parallel processes for each node
        processes = []
        for node in self.nodes:
            processes.append(self.env.process(node.run(self.env)))

        # Run simulation
        self.env.run(until=self.config.simulation_time)

        logger.info(f"Simulation completed at time {self.env.now}")
        summarize_results(self.nodes)

    def get_stats(self) -> Stats:
        """Get statistics for this Simulator and its components"""
        return self.stats.get_stats(components=self.nodes)
