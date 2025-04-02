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
        self.stats = defaultdict(list)

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
        """Load operations from a file (JSON, YAML, or custom format) and organize by tile/core"""
        operations_by_component = defaultdict(list)  # Will store operations by (tile_id, core_id)

        if not os.path.exists(file_path):
            logger.error(f"Operation file {file_path} not found")
            return operations_by_component

        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                elif file_path.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(f)
                else:
                    # Custom parsing logic for other formats
                    lines = f.readlines()
                    data = self._parse_custom_format(lines)

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

    def _parse_custom_format(self, lines):
        """Parse a custom format file into operation data"""
        operations = []
        current_op = {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('OP:'):
                if current_op:
                    operations.append(current_op)
                current_op = {'type': line[3:].strip().lower()}
            elif ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Convert value to appropriate type
                if value.isdigit():
                    value = int(value)
                elif value.startswith('[') and value.endswith(']'):
                    # Parse list of integers
                    value = [int(x.strip()) for x in value[1:-1].split(',')]

                current_op[key] = value

        if current_op:
            operations.append(current_op)

        return operations

    def execute_load(self, op: Load):
        """Process to execute a Load operation"""
        start_time = self.env.now
        logger.info(f"Time {start_time}: Load operation on tile {op.tile} core {op.core} starting")

        # Find the corresponding components in our architecture
        node_id = 0  # Default to first node if not specified in op
        node = self.nodes[node_id]
        tile = node.tiles[op.tile]
        core = tile.cores[op.core]

        # Execute load operation (sequentially, no resource contention)
        yield self.env.timeout(self.config.load_execution_time)

        # Execute on the actual component
        core.execute_load(op.d1)

        # Update stats
        execution_time = self.config.load_execution_time
        core.update_execution_time('load', execution_time)
        tile.update_stats('load')
        tile.update_execution_time(execution_time)
        node.update_stats('load')
        node.update_execution_time(execution_time)

        end_time = self.env.now
        duration = end_time - start_time
        logger.info(f"Time {end_time}: Load operation on tile {op.tile} core {op.core} completed (duration: {duration})")

        # Record statistics
        self.stats['load_times'].append(duration)
        self.stats['completion_times'].append(end_time)

    def execute_set(self, op: Set):
        """Process to execute a Set operation"""
        start_time = self.env.now
        logger.info(f"Time {start_time}: Set operation on tile {op.tile} core {op.core} starting")

        # Find the corresponding components
        node_id = 0  # Default to first node if not specified in op
        node = self.nodes[node_id]
        tile = node.tiles[op.tile]
        core = tile.cores[op.core]

        # Execute set operation (sequentially)
        yield self.env.timeout(self.config.set_execution_time)

        # Execute on the actual component
        core.execute_set(op.imm)

        # Update stats
        execution_time = self.config.set_execution_time
        core.update_execution_time('set', execution_time)
        tile.update_stats('set')
        tile.update_execution_time(execution_time)
        node.update_stats('set')
        node.update_execution_time(execution_time)

        end_time = self.env.now
        duration = end_time - start_time
        logger.info(f"Time {end_time}: Set operation on tile {op.tile} core {op.core} completed (duration: {duration})")

        # Record statistics
        self.stats['set_times'].append(duration)
        self.stats['completion_times'].append(end_time)

    def execute_alu(self, op: Alu):
        """Process to execute an ALU operation"""
        start_time = self.env.now
        logger.info(f"Time {start_time}: ALU operation {op.opcode} on tile {op.tile} core {op.core} starting")

        # Find the corresponding components
        node_id = 0  # Default to first node if not specified in op
        node = self.nodes[node_id]
        tile = node.tiles[op.tile]
        core = tile.cores[op.core]

        # Execute ALU operation (sequentially)
        yield self.env.timeout(self.config.alu_execution_time)

        # Execute on the actual component
        core.execute_alu(op.opcode)

        # Update stats
        execution_time = self.config.alu_execution_time
        core.update_execution_time('alu', execution_time)
        tile.update_stats('alu')
        tile.update_execution_time(execution_time)
        node.update_stats('alu')
        node.update_execution_time(execution_time)

        end_time = self.env.now
        duration = end_time - start_time
        logger.info(f"Time {end_time}: ALU operation {op.opcode} on tile {op.tile} core {op.core} completed (duration: {duration})")

        # Record statistics
        self.stats['alu_times'].append(duration)
        self.stats['completion_times'].append(end_time)

    def execute_mvm(self, op: MVM):
        """Process to execute an MVM operation"""
        start_time = self.env.now
        logger.info(f"Time {start_time}: MVM operation on tile {op.tile} core {op.core} starting")

        # Find the corresponding components
        node_id = 0  # Default to first node if not specified in op
        node = self.nodes[node_id]
        tile = node.tiles[op.tile]
        core = tile.cores[op.core]

        # Assuming the first IMA is used for MVM operations
        ima_id = 0

        # Execute MVM operation (sequentially)
        # Execution time may depend on xbar size
        execution_time = self.config.mvm_execution_time * (len(op.xbar) / 10 + 1)
        yield self.env.timeout(execution_time)

        # Execute on the actual component
        core.execute_mvm(ima_id, op.xbar)

        # Update stats
        core.update_execution_time('mvm', execution_time)
        tile.update_stats('mvm')
        tile.update_execution_time(execution_time)
        node.update_stats('mvm')
        node.update_execution_time(execution_time)

        end_time = self.env.now
        duration = end_time - start_time
        logger.info(f"Time {end_time}: MVM operation on tile {op.tile} core {op.core} completed (duration: {duration})")

        # Record statistics
        self.stats['mvm_times'].append(duration)
        self.stats['completion_times'].append(end_time)

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
        summarize_results(self.stats, self.nodes)

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
