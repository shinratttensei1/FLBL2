"""UCI HAR (Human Activity Recognition Using Smartphones) data loading.

Dataset: Anguita et al., 2013 — UCI ML Repository #240
- 30 subjects, 6 activities, 50 Hz sampling
- Pre-segmented into 128-sample windows (2.56 s) with 50% overlap
- Train: 21 subjects (1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30)
- Test:  9 subjects  (2,4,9,10,12,13,18,20,24)
- Raw signals: 9 channels (body_acc xyz, body_gyro xyz, total_acc xyz)
- Expected at: data/UCI_HAR/UCI_HAR_Dataset/

FL partitioning: 21 train subjects split across 10 Pis via np.array_split
(no duplication — each Pi gets 2 or 3 subjects, test set is shared and fixed).
"""

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from fl_blockchain_evm.core.constants import NUM_CLASSES, NUM_CHANNELS, SC_NAMES

_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get(
    "FL_DATA_DIR",
    os.path.join(_MODULE_ROOT, "data", "UCI_HAR", "UCI_HAR_Dataset"),
)

_NORM_STATS_FILE = os.path.join(DATA_DIR, ".norm_stats.npz")

_TRAIN_SUBJECTS = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
_TEST_SUBJECTS  = [2, 4, 9, 10, 12, 13, 18, 20, 24]

_SIGNAL_NAMES = [
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

_CACHE: dict = {}
_PART_CACHE: dict = {}


def _load_split(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one UCI HAR split (train or test).

    Returns:
        X:        (N, 9, 128) float32
        y:        (N,)        int32   — 0-indexed labels (0–5)
        subjects: (N,)        int32   — subject IDs (1–30)
    """
    if split in _CACHE:
        return _CACHE[split]

    sig_dir = os.path.join(DATA_DIR, split, "Inertial Signals")
    if not os.path.isdir(sig_dir):
        raise FileNotFoundError(
            f"UCI HAR Inertial Signals not found: {sig_dir}\n"
            f"Expected dataset at: {DATA_DIR}"
        )

    print(f"  [UCI-HAR] Loading {split} split from {DATA_DIR} ...")
    channels = []
    for ch in _SIGNAL_NAMES:
        arr = np.loadtxt(os.path.join(sig_dir, f"{ch}_{split}.txt"), dtype=np.float32)
        channels.append(arr)
    X = np.stack(channels, axis=1)                 # (N, 9, 128)

    y = np.loadtxt(os.path.join(DATA_DIR, split, f"y_{split}.txt"),
                   dtype=np.int32) - 1             # 1–6 → 0–5
    subjects = np.loadtxt(os.path.join(DATA_DIR, split, f"subject_{split}.txt"),
                          dtype=np.int32)

    print(f"  [UCI-HAR] {split}: {len(X)} windows  |  "
          f"subjects: {sorted(np.unique(subjects).tolist())}")
    _CACHE[split] = (X, y, subjects)
    return X, y, subjects


def compute_and_save_norm_stats() -> None:
    """Compute global norm stats from training subjects and save to DATA_DIR."""
    X_tr, _, _ = _load_split("train")
    mu = X_tr.mean(axis=(0, 2), keepdims=True)
    sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    np.savez(_NORM_STATS_FILE, mu=mu, sd=sd)
    print(f"  [UCI-HAR] Saved norm stats → {_NORM_STATS_FILE}  (shape={mu.shape})")


def _balance_ros_rus(X: np.ndarray, y: np.ndarray, beta: float = 1.0):
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    active = counts[counts > 0]
    if len(active) == 0:
        return X, y
    m_l, m_s = int(active.max()), int(active.min())
    target = int(m_s + (m_l - m_s) * beta)
    bX, by = [], []
    for c in range(NUM_CLASSES):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        if len(idx) < target:
            idx = np.random.choice(idx, target, replace=True)
        elif len(idx) > target:
            idx = np.random.choice(idx, target, replace=False)
        bX.append(X[idx])
        by.append(np.full(len(idx), c, dtype=np.int32))
    bX = np.concatenate(bX)
    by = np.concatenate(by)
    perm = np.random.permutation(len(bX))
    return bX[perm], by[perm]


def _augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.8:
        x = x + torch.randn_like(x) * 0.05
    if torch.rand(1).item() < 0.5:
        scale = 0.8 + 0.4 * torch.rand(x.size(0), 1, 1, device=x.device)
        x = x * scale
    if torch.rand(1).item() < 0.5:
        shift = torch.randint(-10, 11, (1,)).item()
        if shift > 0:
            x = F.pad(x[:, :, shift:], (0, shift))
        elif shift < 0:
            x = F.pad(x[:, :, :shift], (-shift, 0))
    return x


def load_data(partition_id: int, num_partitions: int, beta: float = 1.0,
              batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Build train/test DataLoaders for one FL client partition (UCI HAR).

    21 training subjects split across num_partitions via np.array_split —
    each Pi gets 2–3 subjects with no duplication.
    """
    train_subjects = sorted(_TRAIN_SUBJECTS)
    chunks = np.array_split(train_subjects, num_partitions)
    my_subjects = set(int(s) for s in chunks[partition_id])

    part_key = tuple(sorted(my_subjects))
    if part_key not in _PART_CACHE:
        X_tr, y_tr, s_tr = _load_split("train")
        X_te, y_te, _    = _load_split("test")

        if os.path.exists(_NORM_STATS_FILE):
            nstats = np.load(_NORM_STATS_FILE)
            mu, sd = nstats["mu"], nstats["sd"]
            print(f"  [UCI-HAR] Normalization: global (pre-saved stats)")
        else:
            mu = X_tr.mean(axis=(0, 2), keepdims=True)
            sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
            print(f"  [UCI-HAR] Normalization: computed on-the-fly")

        my_idx = np.where(np.isin(s_tr, list(my_subjects)))[0]
        X_part = X_tr[my_idx]
        y_part = y_tr[my_idx]
        print(f"  [UCI-HAR] Partition {partition_id}: "
              f"subjects={sorted(my_subjects)}, windows={len(X_part)}")

        if len(X_part) == 0:
            raise ValueError(
                f"No training windows for partition {partition_id} "
                f"(subjects={sorted(my_subjects)})."
            )
        _PART_CACHE[part_key] = (X_part, y_part, X_te, y_te, mu, sd)

    X_part, y_part, X_te, y_te, mu, sd = _PART_CACHE[part_key]

    X_part_n = (X_part - mu) / sd
    X_te_n   = (X_te   - mu) / sd

    if beta > 0:
        X_part_n, y_part = _balance_ros_rus(X_part_n, y_part, beta=beta)

    y_part_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_part]
    y_te_oh   = np.eye(NUM_CLASSES, dtype=np.float32)[y_te]

    X_tr_t = torch.tensor(X_part_n, dtype=torch.float32)
    y_tr_t = torch.tensor(y_part_oh, dtype=torch.float32)
    X_te_t = torch.tensor(X_te_n,   dtype=torch.float32)
    y_te_t = torch.tensor(y_te_oh,  dtype=torch.float32)

    primary_classes = y_tr_t.argmax(dim=1).numpy()
    class_counts    = np.bincount(primary_classes, minlength=NUM_CLASSES).astype(float)
    class_counts    = np.maximum(class_counts, 1.0)
    sample_weights  = 1.0 / class_counts[primary_classes]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return (
        DataLoader(TensorDataset(X_tr_t, y_tr_t),
                   batch_size=batch_size, sampler=sampler, drop_last=True),
        DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size),
    )
