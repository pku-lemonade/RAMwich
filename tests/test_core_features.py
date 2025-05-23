import logging

import numpy as np
import simpy

from ramwich.config import Config
from ramwich.node import Node
from ramwich.ops import MVM, VFU, Copy, Hlt, Load, Set, Store

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def test_core_features():
    # Create configuration
    config = Config(
        num_tiles_per_node=3,
        num_cores_per_tile=2,
        num_mvmus_per_core=1,
        data_width=8,
    )

    # Create SimPy environment
    env = simpy.Environment()

    # Create Node
    node = Node(id=0, config=config)

    # Get tile
    tile = node.get_tile(2)

    # Get cores
    core0 = tile.get_core(0)
    core1 = tile.get_core(1)

    # load test operations
    load_test_ops(core0, core1)

    # load and get test weights
    mat0, mat1 = load_test_weights(core0, core1)

    # Create a test activation
    activation = np.random.randint(0, 7, size=(128,), dtype=np.int32)

    # Add the activation to the tile eDRAM and mark it as valid
    tile.edram.write(0, activation)
    tile.dram_controller.valid[0:128] = True

    # Run the test process
    start_time = env.now

    env.process(tile.run(env))
    env.run()

    end_time = env.now

    # Check the results
    output = tile.edram.cells[256:384].copy()
    expected_output = np.maximum(0, np.dot(mat1, np.maximum(0, np.dot(mat0, activation))))
    print(f"Output: {output}")
    print(f"Expected Output: {expected_output}")
    print(f"Execution Time: {end_time - start_time} cycles")
    assert np.array_equal(output, expected_output), f"Output mismatch: {output} != {expected_output}"
    print("Test passed: Output matches expected result.")


def load_test_ops(core0, core1):
    # Load test operations into core0 and core1
    core0.operations = [
        Set(tile=2, core=0, dest=256, imm=0, vec=1, is_address=True),
        Load(tile=2, core=0, dest=256, read=256, width=16, vec=8),
        Copy(tile=2, core=0, dest=0, read=256, vec=128),
        MVM(tile=2, core=0, xbar=[0]),
        VFU(tile=2, core=0, opcode="relu", dest=256, read_1=128, vec=128),
        Set(tile=2, core=0, dest=384, imm=128, vec=1, is_address=True),
        Store(tile=2, core=0, dest=384, read=256, width=16, vec=8),
        Hlt(tile=2, core=0),
    ]

    core1.operations = [
        Set(tile=2, core=1, dest=256, imm=128, vec=1, is_address=True),
        Load(tile=2, core=1, dest=256, read=256, width=16, vec=8),
        Copy(tile=2, core=1, dest=0, read=256, vec=128),
        MVM(tile=2, core=1, xbar=[0]),
        VFU(tile=2, core=1, opcode="relu", dest=256, read_1=128, vec=128),
        Set(tile=2, core=1, dest=384, imm=256, vec=1, is_address=True),
        Store(tile=2, core=1, dest=384, read=256, width=16, vec=8),
        Hlt(tile=2, core=1),
    ]


def load_test_weights(core0, core1):
    # Load test weights into core0 and core1
    weight0 = np.random.randint(-3, 3, size=(128, 128), dtype=np.int32)
    weight1 = np.random.randint(-3, 3, size=(128, 128), dtype=np.int32)
    core0.get_mvmu(0).load_weights(weight0)
    core1.get_mvmu(0).load_weights(weight1)

    return weight0, weight1


if __name__ == "__main__":
    test_core_features()
