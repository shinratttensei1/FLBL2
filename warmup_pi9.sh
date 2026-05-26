#!/usr/bin/env bash
ping -c 2 raspberrypi9.local && \
ssh pi@raspberrypi9.local \
    "cd ~/FL-Blockchain-EVM && source venv/bin/activate && \
    FL_DATA_DIR=~/FL-Blockchain-EVM/data/PAMAP2/Protocol \
    python3 -c '
from fl_blockchain_evm.core.data import _get_data
_get_data()
print(\"Cache ready.\")
'" && echo "Pi9 cache DONE" || echo "Pi9 FAILED"
