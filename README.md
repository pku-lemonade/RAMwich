# RAMwich

RAMwich is a simulator for heterogeneous RRAM and SRAM CiM architectures.

## TODO

ETA: 2 days
- [ ] convert puma compiler output .puma to .json files that conform to ops.py definitions
- [ ] convert puma simulator config file to .yaml file that can be read by RAMwich
- [ ] test the RAMwich simulator, verify that it can load operations and build architecture

ETA: 2 days
- [ ] convert puma config .yaml files that can be read by RAMwich
- [ ] modify stats class according to puma simulator

ETA: 1 week (1 day for each unit)
- [ ] move the puma simulator functional simultion logic to RAMwich (add details in ima/adc/alu, need to decompose this task into smaller tasks later)
- [ ] test the RAMwich simulator with mlp, verify that cylces, energy, and area are aligned with puma

ETA: 2 days
- [ ] test the RAMwich simulator with mlp, verify that accuracy is aligned
