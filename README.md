# RAMwich

RAMwich is a simulator for heterogeneous RRAM and SRAM CiM architectures.

## How to run

```shell
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python run.py --config <config_file> --ops <ops_file> --weight <weight_file> --activation <activation_file>
```

## Test

To test loading operations and weights:

```shell
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python tests/test_load.py
```

To test MVMU:

```shell
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python tests/test_mvm.py
```

To test DRAM controller:

```shell
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m pytest tests/test_dram_controller.py
```

To test all core features:

```shell
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python tests/test_core_features.py
```

To test all core and tile features:

```shell
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python tests/test_tile_features.py
```

To test a single MLP run on MNIST:

```shell
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python tests/test_mlp_on_mnist.py
```


## TODO

### ETA: 2 days

- [x] convert puma simulator config file to .yaml file that can be read by RAMwich
- [x] convert puma compiler output .puma to .json files that conform to ops.py definitions/load_operations
- [x] test the RAMwich simulator, verify that it can load operations and build architecture

### ETA: 2 days

- [x] convert puma config .yaml files that can be read by RAMwich/dac_config adc_config
- [x] convert puma compiler output .weight to .json files that conform to load_weights
- [ ] modify stats class according to puma simulator

### ETA: 1 week (1 day for each unit)

- [ ] move the puma simulator functional simultion logic to RAMwich (add details in mvmu/adc/alu, need to decompose this task into smaller tasks later) Here is a more detailed plan:
  - [x] implement all core visitor methods
    - [x] implement and test MVMU components(xbar, adc, dac, sna etc.)
    - [x] implement mvm method in MVMU for visitor
    - [x] implement and test core components for calculation(cache, alu)
    - [x] implement visitor method for set, copy, mvm and vfu
    - [x] implement and test core and tile components for load and store(dram, dram controller etc.)
    - [x] implement visitor method for load and store
  - [x] implement all tile visitor methods (send and receive)
  - [ ] test the run time, make sure it is faster than original simulator.
  - [x] run timing simulation, check with puma that cycles match
  - [x] run functional simulation, verify that accuracy match
- [x] test the RAMwich simulator with mlp, verify that cycles, energy, and area are aligned with puma

### ETA: 2 days

- [x] test the RAMwich simulator with mlp, verify that accuracy is aligned
- [ ] Add multi-batch function

### ETA

- [ ] Add SRAM CIM support for inference (2 to 3 days)
- [ ] Verify all parameters of blocks
- [ ] Run test on LeNet-5, ResNet-20, parallel-CNN, DS-CNN

### misc

- [x] save/load weight in npy
- [ ] recalculate MVM latency
- [ ] redesign receive logic
- [ ] Do rusults visualize
