import argparse
import json
import logging
import os
from typing import Dict, List, Any

import numpy as np
import simpy
import yaml

from .config import Config, ADCConfig, DACConfig, NOCConfig, IMAConfig
from .op import Op, Load, Set, Alu, MVM
from .tile import Tile
from .core import Core
from .ima import IMA
from .node import Node
from .visualize import summarize_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class RAMwichSimulator:
    def __init__(self, config_file=None):
        # Default configuration
        self.config = Config()

        # Load configuration if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                if config_file.endswith('.json'):
                    self.config = Config.parse_obj(json.load(f))
                elif config_file.endswith(('.yaml', '.yml')):
                    self.config = Config.parse_obj(yaml.safe_load(f))

        self.env = simpy.Environment()

        # Build the hierarchical architecture
        self.nodes = self._build_architecture()

    def _build_architecture(self):
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

    def load_operations(self, file_path):
        """Load operations from a JSON file and organize by node/tile/core hierarchy"""
        if not os.path.exists(file_path):
            logger.error(f"Operation file {file_path} not found")
            return

        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                else:
                    logger.error(f"Unsupported file format: {file_path}. Only JSON is supported.")
                    return

            # Convert raw data to operation objects and organize by node/tile/core
            for op_data in data:
                op_type = op_data.pop('type', None)
                node_id = op_data.get('node', 0)
                tile_id = op_data.get('tile', 0)
                core_id = op_data.get('core', 0)

                if node_id >= len(self.nodes):
                    logger.warning(f"Operation specifies node {node_id} but only {len(self.nodes)} nodes exist")
                    continue

                node = self.nodes[node_id]

                try:
                    tile = node.get_tile(tile_id)
                    core = tile.get_core(core_id)

                    # Create the appropriate operation object
                    op = None
                    if op_type == 'load':
                        op = Load(**op_data)
                    elif op_type == 'set':
                        op = Set(**op_data)
                    elif op_type == 'alu':
                        op = Alu(**op_data)
                    elif op_type == 'mvm':
                        op = MVM(**op_data)
                    else:
                        logger.warning(f"Unknown operation type: {op_type}")
                        continue

                    # Store the operation in the core
                    if not hasattr(core, 'operations'):
                        core.operations = []
                    core.operations.append(op)

                except ValueError as e:
                    logger.warning(str(e))

        except Exception as e:
            logger.error(f"Error loading operations: {e}")

    def execute_load(self, op: Load):
        """Process to execute a Load operation"""
        # Find the corresponding components in our architecture
        node = self.nodes[op.node]
        tile = node.tiles[op.tile]
        core = tile.cores[op.core]

        # Execute load operation (sequentially, no resource contention)
        yield self.env.timeout(self.config.load_execution_time)

        # Execute on the actual component
        core.execute_load(op.d1)

    def execute_set(self, op: Set):
        """Process to execute a Set operation"""
        # Find the corresponding components
        node = self.nodes[op.node]
        tile = node.tiles[op.tile]
        core = tile.cores[op.core]

        # Execute set operation (sequentially)
        yield self.env.timeout(self.config.set_execution_time)

        # Execute on the actual component
        core.execute_set(op.imm)

    def execute_alu(self, op: Alu):
        """Process to execute an ALU operation"""
        # Find the corresponding components
        node = self.nodes[op.node]
        tile = node.tiles[op.tile]
        core = tile.cores[op.core]

        # Execute ALU operation (sequentially)
        yield self.env.timeout(self.config.alu_execution_time)

        # Execute on the actual component
        core.execute_alu(op.opcode)

    def execute_mvm(self, op: MVM):
        """Process to execute an MVM operation"""
        # Find the corresponding components
        node = self.nodes[op.node]
        tile = node.tiles[op.tile]
        core = tile.cores[op.core]

        # Execute MVM operation (sequentially)
        # Execution time may depend on xbar size
        execution_time = self.config.mvm_execution_time * (len(op.xbar) / 10 + 1)
        yield self.env.timeout(execution_time)

        # Execute on the actual component
        core.execute_mvm(op.ima, op.xbar)

    def run_simulation(self, op_file):
        """Run the simulation with operations from the specified file"""
        # Load operations into node/tile/core hierarchy
        self.load_operations(op_file)

        # Create and schedule parallel processes for each node
        node_processes = []
        for node in self.nodes:
            node_processes.append(self.env.process(node.run(self, self.env)))

        # Run simulation
        self.env.run(until=self.config.simulation_time)

        logger.info(f"Simulation completed at time {self.env.now}")
        summarize_results(self.nodes)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='RAMwich Simulator')
    parser.add_argument('op_file', help='File containing operations to execute')
    parser.add_argument('--config', help='Configuration file (JSON or YAML)')
    args = parser.parse_args()

    # Create and run simulator
    simulator = RAMwichSimulator(config_file=args.config)
    simulator.run_simulation(args.op_file)

if __name__ == "__main__":
    main()
