# ChangeLog

All notable changes to PyGOP will be documented here.

## [0.2.3] 2020-09-15

### Added
- recommendation to install tensorflow 1.14.0

### Fixed
- fixing keras version to 2.2.4 in the requirement and setup


## [0.2.2] 2019-04-24

### Added

- Test Suite: unit test, PEP8 verification
- Continuous Integration (Travis CI)
- Mnist Example
- Instructions on Contributions
- Improved Documentation
- requirements.txt
- Functionality to specify optimizer's parameters

### Fixed

- learning rate update in utility/gop_utils.network_trainer()
- improve coding style according to PEP8

## [0.2.1] 2018-12-29

### Added
- new wheel for version 0.2.1 in releases

### Fixed
- move the tmp files removal step from progressive_learn to the end of fit() in all models (hemlgop.py, homlgop.py, hemlrn.py homlrn.py, pop.py, popfast.py, popmemo.py, popmemh.py). This fixes the bug that removes train_states.pickle before finetuning. 
- change file opening option from 'r' to 'rb' in utility/misc.initialize_states() to read train_states.pickle in Python3

## [0.2.0] 2018-12-17

### Added
- utility/block_update.py
- utility/calculate_memory.py
- Added CHANGELOG.md to keep track of major changes
- Added releases directory to keep track of different wheel versions.
- Functionalities to spawn new process when calculating memory block (utility/gop_utils.calculate_memory_block_standalone() and utility/calculate_memory.py). This prevents potential OOM errors when tensorflow-gpu does not release memory right after the block finishes. 
- Functionalities to spawn new process when finetuning some blocks (utility/gop_utils.block_update_standalone() and utility/block_update.py). This also prevents potential OOM errors mentioned above.

### Fixed
- utility/misc.check_model_parameters()
- models/_model.print_performance()  
