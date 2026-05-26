"""Shared constants — selected at import time via FL_DATASET env var.

  FL_DATASET=ucihar  (default) — UCI HAR, 6 classes, 9 channels, 128-sample windows
  FL_DATASET=pamap2             — PAMAP2, 12 classes, 27 channels, 512-sample windows
"""

import os
from typing import List

_DATASET = os.getenv("FL_DATASET", "ucihar")

if _DATASET == "pamap2":
    DATASET_NAME: str = "PAMAP2"
    ACTIVITY_NAMES: List[str] = [
        "LYING", "SITTING", "STANDING", "WALKING", "RUNNING", "CYCLING",
        "NORDIC_WALKING", "ASCENDING_STAIRS", "DESCENDING_STAIRS",
        "VACUUM_CLEANING", "IRONING", "ROPE_JUMPING",
    ]
    NUM_CLASSES  = 12
    NUM_CHANNELS = 27
    WINDOW_SIZE  = 512
    WINDOW_STEP  = 256
else:
    # UCI HAR (default)
    DATASET_NAME: str = "UCI HAR"
    ACTIVITY_NAMES: List[str] = [
        "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
        "SITTING", "STANDING", "LAYING",
    ]
    NUM_CLASSES  = 6
    NUM_CHANNELS = 9
    WINDOW_SIZE  = 128
    WINDOW_STEP  = 64

SC_NAMES: List[str] = ACTIVITY_NAMES
