.. _changelog:

*************
Change Log
*************

All notable changes to PyGOP will be documented here.

[0.2.0] 2018-12-17
==================

Added
-----

- utility/block_update.py
- utility/calculate_memory.py
- Added CHANGELOG.md to keep track of major changes
- Added releases directory to keep track of different wheel versions.
- Functionalities to spawn new process when calculating memory block (utility/gop_utils.calculate_memory_block_standalone() and utility/calculate_memory.py). This prevents potential OOM errors when tensorflow-gpu does not release memory right after the block finishes.
- Functionalities to spawn new process when finetuning some blocks (utility/gop_utils.block_update_standalone() and utility/block_update.py). This also prevents potential OOM errors mentioned above.

Fixed
-----
- utility/misc.check_model_parameters()
- models/_model.print_performance()

