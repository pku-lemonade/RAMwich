import logging

from ramwich import RAMwich
from ramwich.ops import Send

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    config_file = "examples/mlp_l4_mnist/config.yaml"
    ops_file = "examples/mlp_l4_mnist/ops.json"
    weights_file = "examples/mlp_l4_mnist/weights.npz"

    simulator = RAMwich(config_file=config_file)
    simulator.load_operations(file_path=ops_file)
    simulator.load_weights(file_path=weights_file)

    validate_op = Send(type="send", node=0, tile=0, mem_addr=768, target_tile=3, width=16, vec=1)
    assert simulator.get_node(0).get_tile(0).operations[0] == validate_op, "Operation not loaded correctly"
    assert (
        simulator.get_node(0).get_tile(2).cores[0].mvmus[0].rram_xbar_array.neg_xbar[0][0][0] == 1
    ), "Weight not loaded correctly"
    print("Operation and weight loaded correctly.")


if __name__ == "__main__":
    main()
