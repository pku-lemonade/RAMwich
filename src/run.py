import simpy
import json
import yaml
import logging
import numpy as np
from collections import defaultdict
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from op import Op, Load, Set, Alu, MVM
from tile import Tile
from ima import IMA
from node import Node
from visualize import summarize_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config(BaseModel):
    """Configuration for the RAMwich Simulator"""
    num_nodes: int = Field(default=1, description="Number of nodes in the system")
    num_tiles_per_node: int = Field(default=4, description="Number of tiles per node")
    num_cores_per_tile: int = Field(default=2, description="Number of cores per tile")
    num_imas_per_core: int = Field(default=2, description="Number of IMAs per core")
    num_xbars_per_ima: int = Field(default=4, description="Number of crossbars per IMA")
    alu_execution_time: int = Field(default=2, description="Execution time for ALU operations")
    load_execution_time: int = Field(default=5, description="Execution time for Load operations")
    set_execution_time: int = Field(default=1, description="Execution time for Set operations")
    mvm_execution_time: int = Field(default=10, description="Execution time for MVM operations")
    simulation_time: int = Field(default=1000, description="Maximum simulation time")

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
        """Load operations from a JSON file and organize by tile/core"""
        operations_by_component = defaultdict(list)  # Will store operations by (tile_id, core_id)

        if not os.path.exists(file_path):
            logger.error(f"Operation file {file_path} not found")
            return operations_by_component

        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                else:
                    logger.error(f"Unsupported file format: {file_path}. Only JSON is supported.")
                    return operations_by_component

            # Convert raw data to operation objects and organize by tile/core
            for op_data in data:
                op_type = op_data.pop('type', None)

                # Extract tile and core information to use as keys
                tile_id = op_data.get('tile', 0)
                core_id = op_data.get('core', 0)
                component_key = (tile_id, core_id)

                if op_type == 'load':
                    operations_by_component[component_key].append(Load(**op_data))
                elif op_type == 'set':
                    operations_by_component[component_key].append(Set(**op_data))
                elif op_type == 'alu':
                    operations_by_component[component_key].append(Alu(**op_data))
                elif op_type == 'mvm':
                    operations_by_component[component_key].append(MVM(**op_data))
                else:
                    logger.warning(f"Unknown operation type: {op_type}")

        except Exception as e:
            logger.error(f"Error loading operations: {e}")

        return operations_by_component

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
        core.execute_mvm(ima_id, op.xbar)

    def run_component_operations(self, tile_id, core_id, operations):
        """Execute a sequence of operations for a specific tile/core combination"""
        logger.info(f"Starting operation execution for tile {tile_id}, core {core_id} with {len(operations)} operations")

        for op in operations:
            if isinstance(op, Load):
                yield self.env.process(self.execute_load(op))
            elif isinstance(op, Set):
                yield self.env.process(self.execute_set(op))
            elif isinstance(op, Alu):
                yield self.env.process(self.execute_alu(op))
            elif isinstance(op, MVM):
                yield self.env.process(self.execute_mvm(op))
            else:
                logger.warning(f"Unknown operation type: {type(op)}")

        logger.info(f"Completed all operations for tile {tile_id}, core {core_id}")

    def run_simulation(self, op_file):
        """Run the simulation with operations from the specified file"""
        operations_by_component = self.load_operations(op_file)

        if not operations_by_component:
            logger.error("No operations to simulate")
            return

        total_operations = sum(len(ops) for ops in operations_by_component.values())
        logger.info(f"Starting simulation with {total_operations} operations across {len(operations_by_component)} tile/core combinations")

        # Create and schedule parallel processes for each tile/core combination
        for (tile_id, core_id), operations in operations_by_component.items():
            if operations:  # Only create processes for components with operations
                self.env.process(self.run_component_operations(tile_id, core_id, operations))

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
