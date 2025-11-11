"""
Debug monitoring utilities for detecting and diagnosing stuck simulations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import simpy

if TYPE_CHECKING:
    from .ramwich import RAMwich

logger = logging.getLogger(__name__)


class SimulationMonitor:
    """Monitor simulation progress and detect stuck states."""

    def __init__(self, simulator: "RAMwich", timeout: int = 100000, save_time: int = None, debug_file: str = None):
        """
        Initialize the simulation monitor.

        Args:
            simulator: The RAMwich simulator instance to monitor
            timeout: Maximum simulation time before considering it stuck (in cycles)
            save_time: Specific time to save debug info (if None, only saves on timeout)
            debug_file: Path to save debug information (default: debug_output_<timestamp>.txt)
        """
        self.simulator = simulator
        self.timeout = timeout
        self.save_time = save_time
        self.last_progress_time = 0
        self.check_interval = max(1000, timeout // 100)  # Check every 1% of timeout
        self.debug_saved = False  # Track if debug info has been saved

        # Setup debug file
        if debug_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = f"debug_output_{timestamp}.txt"
        self.debug_file = Path(debug_file)

        # Track operation execution
        self.operation_tracker = OperationTracker()
        self._setup_operation_tracking()

    def _setup_operation_tracking(self):
        """Setup tracking for all operations in the simulator."""
        for node in self.simulator.nodes:
            for tile in node.tiles:
                # Track tile operations
                for i, op in enumerate(tile.operations):
                    self.operation_tracker.register_operation(
                        f"node{node.id}_tile{tile.id}_op{i}", op, "tile", node.id, tile.id
                    )

                # Track core operations
                for core in tile.cores:
                    for i, op in enumerate(core.operations):
                        self.operation_tracker.register_operation(
                            f"node{node.id}_tile{tile.id}_core{core.id}_op{i}", op, "core", node.id, tile.id, core.id
                        )

    def monitor_process(self, env: simpy.Environment):
        """
        SimPy process that monitors simulation progress.
        Saves debug information to file at specified time or on timeout.
        """
        while True:
            yield env.timeout(self.check_interval)

            current_time = env.now

            # Check if we should save debug info at the specified time
            if self.save_time is not None and current_time >= self.save_time and not self.debug_saved:
                logger.info(f"Saving debug state at time {current_time}")
                self.save_debug_state(env, is_timeout=False)
                self.debug_saved = True

            # Check if we've exceeded the timeout
            if current_time >= self.timeout:
                logger.error(f"Simulation appears stuck at time {current_time}")
                self.save_debug_state(env, is_timeout=True)
                raise RuntimeError(f"Simulation timeout at cycle {current_time}. Debug info saved to {self.debug_file}")

    def save_debug_state(self, env: simpy.Environment, is_timeout: bool = False):
        """Save detailed debug information about the current simulation state to file."""
        with open(self.debug_file, "w") as f:
            self._write_line(f, "=" * 80)
            if is_timeout:
                self._write_line(f, f"SIMULATION STUCK DETECTED AT TIME {env.now}")
            else:
                self._write_line(f, f"DEBUG STATE SNAPSHOT AT TIME {env.now}")
            self._write_line(f, f"Debug output generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._write_line(f, "=" * 80)
            self._write_line(f, "")

            # Write last executed operation summary
            self._write_last_executed_op_summary(f)

            # Write first un-executed operation summary
            self._write_first_unexecuted_op_summary(f)

            # Write operation execution summary
            self._write_operation_summary(f, env)

            # Write detailed state for each node
            for node in self.simulator.nodes:
                self._write_line(f, "\n" + "=" * 80)
                self._write_line(f, f"NODE {node.id} STATE:")
                self._write_line(f, "=" * 80)

                for tile in node.tiles:
                    self._write_tile_state(f, tile, env)

        logger.error(f"Debug information saved to {self.debug_file}")

    def _write_line(self, f, text):
        """Write a line to the debug file."""
        f.write(text + "\n")

    def _write_last_executed_op_summary(self, f):
        """Write a summary of the last executed operation for each tile and core."""
        self._write_line(f, "\n" + "=" * 80)
        self._write_line(f, "LAST EXECUTED OPERATION SUMMARY")
        self._write_line(f, "=" * 80)

        last_tile_ops = {}  # (node_id, tile_id) -> op_info
        last_core_ops = {}  # (node_id, tile_id, core_id) -> op_info

        for _op_id, info in self.operation_tracker.operations.items():
            if info["executed"]:
                level = info["level"]
                node_id = info["node_id"]
                tile_id = info["tile_id"]
                core_id = info["core_id"]
                exec_time = info["execution_time"]

                if level == "tile":
                    key = (node_id, tile_id)
                    if key not in last_tile_ops or exec_time > last_tile_ops[key]["execution_time"]:
                        last_tile_ops[key] = info
                elif level == "core":
                    key = (node_id, tile_id, core_id)
                    if key not in last_core_ops or exec_time > last_core_ops[key]["execution_time"]:
                        last_core_ops[key] = info

        self._write_line(f, "\nLast Executed Tile Operations:")
        self._write_line(f, "-" * 30)
        if not last_tile_ops:
            self._write_line(f, "  None")
        else:
            for (node_id, tile_id), info in sorted(last_tile_ops.items()):
                self._write_line(f, f"  Node {node_id}, Tile {tile_id}:")
                self._write_line(f, f"    Time: {info['execution_time']}")
                self._write_line(f, f"    Operation: {info['operation']}")

        self._write_line(f, "\nLast Executed Core Operations:")
        self._write_line(f, "-" * 30)
        if not last_core_ops:
            self._write_line(f, "  None")
        else:
            for (node_id, tile_id, core_id), info in sorted(last_core_ops.items()):
                self._write_line(f, f"  Node {node_id}, Tile {tile_id}, Core {core_id}:")
                self._write_line(f, f"    Time: {info['execution_time']}")
                self._write_line(f, f"    Operation: {info['operation']}")
        self._write_line(f, "")

    def _write_first_unexecuted_op_summary(self, f):
        """Write a summary of the first un-executed operation for each tile and core."""
        self._write_line(f, "\n" + "=" * 80)
        self._write_line(f, "FIRST UN-EXECUTED OPERATION SUMMARY")
        self._write_line(f, "=" * 80)

        first_unexecuted_tile_ops = {}
        first_unexecuted_core_ops = {}

        # Sort operations by parsing the IDs to ensure correct numerical order
        def sort_key(op_id):
            parts = op_id.replace("node", "").replace("tile", "_").replace("core", "_").replace("op", "_").split("_")
            return [int(p) for p in parts if p]

        sorted_op_ids = sorted(self.operation_tracker.operations.keys(), key=sort_key)

        for op_id in sorted_op_ids:
            info = self.operation_tracker.operations[op_id]
            if not info["executed"]:
                level = info["level"]
                node_id = info["node_id"]
                tile_id = info["tile_id"]
                core_id = info["core_id"]

                if level == "tile":
                    key = (node_id, tile_id)
                    if key not in first_unexecuted_tile_ops:
                        first_unexecuted_tile_ops[key] = info
                elif level == "core":
                    key = (node_id, tile_id, core_id)
                    if key not in first_unexecuted_core_ops:
                        first_unexecuted_core_ops[key] = info

        self._write_line(f, "\nFirst Un-executed Tile Operations:")
        self._write_line(f, "-" * 30)
        if not first_unexecuted_tile_ops:
            self._write_line(f, "  All tile operations executed or none found.")
        else:
            for (node_id, tile_id), info in sorted(first_unexecuted_tile_ops.items()):
                self._write_line(f, f"  Node {node_id}, Tile {tile_id}:")
                self._write_line(f, f"    Operation: {info['operation']}")

        self._write_line(f, "\nFirst Un-executed Core Operations:")
        self._write_line(f, "-" * 30)
        if not first_unexecuted_core_ops:
            self._write_line(f, "  All core operations executed or none found.")
        else:
            for (node_id, tile_id, core_id), info in sorted(first_unexecuted_core_ops.items()):
                self._write_line(f, f"  Node {node_id}, Tile {tile_id}, Core {core_id}:")
                self._write_line(f, f"    Operation: {info['operation']}")
        self._write_line(f, "")

    def _write_operation_summary(self, f, env):
        """Write summary of executed vs non-executed operations."""
        self._write_line(f, "\n" + "=" * 80)
        self._write_line(f, "OPERATION EXECUTION SUMMARY")
        self._write_line(f, "=" * 80)

        executed, not_executed = self.operation_tracker.get_execution_summary()

        self._write_line(f, f"\nTotal operations: {executed + not_executed}")
        self._write_line(f, f"Executed operations: {executed}")
        self._write_line(f, f"Not executed operations: {not_executed}")
        self._write_line(f, f"Execution rate: {executed / (executed + not_executed) * 100:.2f}%")

        # Write details of non-executed operations
        if not_executed > 0:
            self._write_line(f, "\n" + "-" * 80)
            self._write_line(f, "NON-EXECUTED OPERATIONS:")
            self._write_line(f, "-" * 80)

            for op_id, info in self.operation_tracker.operations.items():
                if not info["executed"]:
                    self._write_line(f, f"\n  {op_id}:")
                    self._write_line(f, f"    Type: {info['type']}")
                    self._write_line(f, f"    Level: {info['level']}")
                    self._write_line(
                        f,
                        f"    Location: Node {info['node_id']}, Tile {info['tile_id']}"
                        + (f", Core {info['core_id']}" if info["core_id"] is not None else ""),
                    )
                    self._write_line(f, f"    Operation: {info['operation']}")

        # Write details of executed operations
        self._write_line(f, "\n" + "-" * 80)
        self._write_line(f, "EXECUTED OPERATIONS:")
        self._write_line(f, "-" * 80)

        for op_id, info in self.operation_tracker.operations.items():
            if info["executed"]:
                self._write_line(f, f"\n  {op_id}:")
                self._write_line(f, f"    Type: {info['type']}")
                self._write_line(f, f"    Level: {info['level']}")
                self._write_line(
                    f,
                    f"    Location: Node {info['node_id']}, Tile {info['tile_id']}"
                    + (f", Core {info['core_id']}" if info["core_id"] is not None else ""),
                )
                self._write_line(f, f"    Execution time: {info['execution_time']}")
                self._write_line(f, f"    Operation: {info['operation']}")

    def _write_tile_state(self, f, tile, env):
        """Write the state of a single tile to file."""
        self._write_line(f, f"\n  TILE {tile.id}:")
        self._write_line(f, f"    Start time: {tile.start_time}")
        self._write_line(f, f"    Current time: {env.now}")
        self._write_line(f, f"    Active cycles: {tile.active_cycles}")
        self._write_line(f, f"    Total operations: {len(tile.operations)}")

        # Write tile operations status
        if tile.operations:
            self._write_line(f, "    Tile operations:")
            for i, op in enumerate(tile.operations):
                op_id = f"node{tile.parent.id}_tile{tile.id}_op{i}"
                executed = self.operation_tracker.is_executed(op_id)
                status = "✓ EXECUTED" if executed else "✗ NOT EXECUTED"
                self._write_line(f, f"      [{i}] {status} - {op.type}: {op}")

        # Write DRAM controller state
        self._write_line(f, "    DRAM Controller:")
        self._write_line(f, f"      Running: {tile.dram_controller.is_running}")
        self._write_line(f, f"      Pending requests: {len(tile.dram_controller.requests.items)}")
        self._write_line(f, f"      Valid entries: {tile.dram_controller.valid.sum()}")

        # Write router state
        self._write_line(f, "    Router:")
        self._write_line(f, f"      Running: {tile.router.is_running}")
        self._write_line(f, f"      Send queue size: {len(tile.router.send_queue.items)}")
        self._write_line(f, f"      Packets sent: {tile.router.stats.packets_sent}")
        self._write_line(f, f"      Packets received: {tile.router.stats.packets_received}")

        # Write receive buffers state
        if hasattr(tile.router, "receive_buffers"):
            non_empty_buffers = []
            for source_id, buffer in tile.router.receive_buffers.items():
                if len(buffer.items) > 0:
                    non_empty_buffers.append((source_id, len(buffer.items)))
            if non_empty_buffers:
                self._write_line(f, f"      Non-empty receive buffers: {non_empty_buffers}")

        # Write core states
        for core in tile.cores:
            self._write_core_state(f, core, env)

    def _write_core_state(self, f, core, env):
        """Write the state of a single core to file."""
        self._write_line(f, f"\n    CORE {core.id}:")
        self._write_line(f, f"      Start time: {core.start_time}")
        self._write_line(f, f"      Active cycles: {core.active_cycles}")
        self._write_line(f, f"      Total operations: {len(core.operations)}")

        # Write current operation being executed
        if core.operations:
            self._write_line(f, "      Core operations:")
            for i, op in enumerate(core.operations):
                op_id = f"node{core.parent.parent.id}_tile{core.parent.id}_core{core.id}_op{i}"
                executed = self.operation_tracker.is_executed(op_id)
                status = "✓ EXECUTED" if executed else "✗ NOT EXECUTED"
                self._write_line(f, f"        [{i}] {status} - {op.type}: {op}")

        # Write MVMU states
        for mvmu in core.mvmus:
            if mvmu.stats.activation_count > 0:
                self._write_line(f, f"      MVMU {mvmu.id}:")
                self._write_line(f, f"        Activations: {mvmu.stats.activation_count}")
                self._write_line(f, f"        Weights loaded: {mvmu.weights is not None}")

    def mark_operation_executed(self, op_id: str, execution_time: int):
        """Mark an operation as executed."""
        self.operation_tracker.mark_executed(op_id, execution_time)


class OperationTracker:
    """Track execution status of all operations."""

    def __init__(self):
        self.operations = {}

    def register_operation(self, op_id: str, operation, level: str, node_id: int, tile_id: int, core_id: int = None):
        """Register an operation for tracking."""
        self.operations[op_id] = {
            "operation": operation,
            "type": operation.type if hasattr(operation, "type") else str(type(operation).__name__),
            "level": level,  # 'tile' or 'core'
            "node_id": node_id,
            "tile_id": tile_id,
            "core_id": core_id,
            "executed": False,
            "execution_time": None,
        }

    def mark_executed(self, op_id: str, execution_time: int):
        """Mark an operation as executed."""
        if op_id in self.operations:
            self.operations[op_id]["executed"] = True
            self.operations[op_id]["execution_time"] = execution_time

    def is_executed(self, op_id: str) -> bool:
        """Check if an operation has been executed."""
        return self.operations.get(op_id, {}).get("executed", False)

    def get_execution_summary(self):
        """Get summary of executed vs non-executed operations."""
        executed = sum(1 for op in self.operations.values() if op["executed"])
        not_executed = len(self.operations) - executed
        return executed, not_executed
