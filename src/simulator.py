import json
import logging
import os
from typing import List

import simpy
import yaml

from .config import Config
from .op import Op, Load, Set, ALU, MVM, OpType
from .tile import Tile
from .core import Core
from .ima import IMA
from .node import Node
from .visualize import summarize_results

# Configure logging
logger = logging.getLogger(__name__)

class RAMwichSimulator:
    def __init__(self, config_file: str):
        # Default configuration
        self.config: Config = Config()

        # Load configuration from file
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} not found")

        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                self.config = Config.parse_obj(json.load(f))
            elif config_file.endswith(('.yaml', '.yml')):
                self.config = Config.parse_obj(yaml.safe_load(f))
            else:
                raise ValueError(f"Unsupported config format: {config_file}. Use JSON or YAML.")

        self.env = simpy.Environment()
        self.stats = Stat()  # Add stats attribute

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
                    imas = []

                    for ima_id in range(self.config.num_imas_per_core):
                        # Create IMA with its xbars
                        ima = IMA(
                            id=ima_id,
                            num_xbars=self.config.num_xbars_per_ima
                        )
                        imas.append(ima)

                    # Create Core with its IMAs
                    core = Core(
                        id=core_id,
                        imas=imas
                    )
                    cores.append(core)

                # Create Tile with its Cores
                tile = Tile(
                    id=tile_id,
                    cores=cores
                )
                tiles.append(tile)

            # Create Node with its Tiles
            node = Node(
                id=node_id,
                tiles=tiles
            )
            nodes.append(node)

        return nodes

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def load_operations(self, file_path: str):
        """Load operations from a JSON file and organize by node/tile/core hierarchy"""
        if not os.path.exists(file_path):
            logger.error(f"Operation file {file_path} not found")
            return

        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
            else:
                logger.error(f"Unsupported file format: {file_path}. Only JSON is supported.")
                return

        # Convert raw data to operation objects and organize by node/tile/core
        for op_data in data:
            try:
                # Parse the operation using Pydantic discriminated union
                # This automatically creates the correct object type based on the 'type' field
                op = OpType.model_validate(op_data)

                # Access the hierarchical components
                node = self.get_node(op.node)
                tile = node.get_tile(op.tile)
                core = tile.get_core(op.core)

                # Store the operation in the core
                core.operations.append(op)

            except ValueError as e:
                logger.warning(str(e))

    def run(self, ops_file: str):
        """Run the simulation with operations from the specified file"""
        # Load operations into node/tile/core hierarchy
        self.load_operations(ops_file)

        # Create and schedule parallel processes for each node
        processes = []
        for node in self.nodes:
            processes.append(self.env.process(node.run(self, self.env)))

        # Run simulation
        self.env.run(until=self.config.simulation_time)

        logger.info(f"Simulation completed at time {self.env.now}")
        summarize_results(self.nodes)

    def get_stats(self):
        """Get statistics for this Simulator and its components"""
        return self.stats.get_stats(components=self.nodes)
