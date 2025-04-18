# RAMwich

RAMwich is a simulator for heterogeneous RRAM and SRAM CiM architectures.

## Test

```shell
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python run.py --config <config_file> --ops <ops_file> --weight <weight_file>
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
  - [ ] implement all core visitor methods
    - [x] implement and test MVMU components(xbar, adc, dac, sna etc.)
    - [x] implement mvm method in MVMU for visitor
    - [x] implement and test core components for calculation(cache, alu)
    - [x] implement visitor method for set, copy, mvm and vfu
    - [ ] implement and test core and tile components for load and store(dram, dram controller etc.)
    - [ ] implement visitor method for load and store
  - [ ] implement all tile visitor methods
    - [ ] implement noc
    - [ ] implement visitor method for send and receive
  - [ ] run timing simulation, check with puma that cycles match
  - [ ] run functional simulation, verify that accuracy match
- [ ] test the RAMwich simulator with mlp, verify that cylces, energy, and area are aligned with puma

### ETA: 2 days

- [ ] test the RAMwich simulator with mlp, verify that accuracy is aligned

### misc

- [ ] save/load weight in npy
