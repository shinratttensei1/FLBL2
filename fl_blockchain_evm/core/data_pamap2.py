"""PAMAP2 Physical Activity Monitoring data loading, caching, balancing, and augmentation.

Dataset: PAMAP2 — Reiss & Stricker, 2012
- 9 subjects (101-109), 12 standard activity classes + transient (label 0, excluded)
- 54 columns per row (space-separated):
    col 0      : timestamp (s)
    col 1      : activityID (0=transient, standard: 1-7,12,13,16,17,24)
    col 2      : heart rate (bpm; excluded)
    cols 3-15  : IMU hand  — temp, acc16g(x,y,z), acc6g(x,y,z), gyro(x,y,z), mag(x,y,z)
    cols 16-19 : IMU hand  — orientation (always NaN, excluded)
    cols 20-32 : IMU chest — same 13 valid features
    cols 33-36 : IMU chest — orientation (excluded)
    cols 37-49 : IMU ankle — same 13 valid features
    cols 50-53 : IMU ankle — orientation (excluded)
- Sampled at 100 Hz
- Expected at: data/PAMAP2/subject<NNN>.dat

FL partitioning: subjects 101-107 for training (7 clients, one per subject),
subjects 108-109 held out for testing. Partitioned by subject ID (non-IID).

Sliding-window segmentation: 512-sample windows with 256-sample stride
(≈ 5.12 s windows at 50 % overlap — same temporal duration as MHEALTH).
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from fl_blockchain_evm.core.constants import (
    NUM_CLASSES, NUM_CHANNELS, SC_NAMES, WINDOW_SIZE, WINDOW_STEP,
)

# ── Dataset location ──────────────────────────────────────────
_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get(
    "FL_DATA_DIR",
    os.path.join(_MODULE_ROOT, "data", "PAMAP2", "Protocol"),
)

_NPY_CACHE_DIR = os.path.join(DATA_DIR, ".npy_cache")

_TRAIN_SUBJECTS = [101, 102, 103, 104, 105, 106, 107]
_TEST_SUBJECTS  = [108, 109]

_CACHE: dict = {}
_PART_CACHE: dict = {}

_NORM_STATS_FILE = os.path.join(DATA_DIR, ".norm_stats.npz")

# ── PAMAP2 column selection ───────────────────────────────────
# Keep only 3-axis acc (16g) + gyro + mag for each of 3 IMUs (hand/chest/ankle).
# Dropped: heart rate (col 2), temperature (cols 3,20,37), acc 6g (cols 7-9,24-26,41-43).
# Per-IMU layout (13 cols, base B): temp=B, acc16g=B+1..3, acc6g=B+4..6, gyro=B+7..9, mag=B+10..12
_SENSOR_COLS: list = (
    list(range(4,  7))  + list(range(10, 16)) +   # IMU hand:  acc16g + gyro + mag
    list(range(21, 24)) + list(range(27, 33)) +   # IMU chest: acc16g + gyro + mag
    list(range(38, 41)) + list(range(44, 50))     # IMU ankle: acc16g + gyro + mag
)  # 3×(3+6) = 27 channels

# Map raw PAMAP2 activityID → internal 1-indexed label (0 = null / transient)
_PAMAP2_ACT_TO_IDX: dict = {
    1:  1,   # lying
    2:  2,   # sitting
    3:  3,   # standing
    4:  4,   # walking
    5:  5,   # running
    6:  6,   # cycling
    7:  7,   # nordic walking
    12: 8,   # ascending stairs
    13: 9,   # descending stairs
    16: 10,  # vacuum cleaning
    17: 11,  # ironing
    24: 12,  # rope jumping
}


# ── Raw file loading ──────────────────────────────────────────

def _load_subject(subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw sensor data for one PAMAP2 subject.

    Returns:
        data:   (T, 27) float32 -- 27 sensor channels (acc16g + gyro + mag × 3 IMUs)
        labels: (T,)    int32   -- per-sample label (0=null/transient, 1-12=activity)
    """
    cache_data   = os.path.join(_NPY_CACHE_DIR, f"s{subject_id}_data.npy")
    cache_labels = os.path.join(_NPY_CACHE_DIR, f"s{subject_id}_labels.npy")

    if os.path.exists(cache_data) and os.path.exists(cache_labels):
        try:
            d = np.load(cache_data)
            if d.shape[1] != NUM_CHANNELS:
                print(f"  [PAMAP2] Cache shape mismatch for subject {subject_id} "
                      f"({d.shape[1]} ch, expected {NUM_CHANNELS}); rebuilding...")
                raise ValueError("stale cache")
            return d, np.load(cache_labels)
        except Exception:
            for p in (cache_data, cache_labels):
                try:
                    os.remove(p)
                except OSError:
                    pass

    path = os.path.join(DATA_DIR, f"subject{subject_id}.dat")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PAMAP2 subject file not found: {path}\n"
            f"Please place PAMAP2 .dat files in {DATA_DIR}"
        )

    print(f"  [PAMAP2] Parsing subject {subject_id} (first run — building cache)...")
    df = pd.read_csv(path, header=None, sep=r"\s+", dtype=np.float64)

    # Remap activity IDs: standard activities → 1-12, everything else → 0
    raw_act = df.iloc[:, 1].fillna(0).astype(np.int32)
    labels  = np.array([_PAMAP2_ACT_TO_IDX.get(int(a), 0) for a in raw_act],
                       dtype=np.int32)

    # Extract valid sensor columns and interpolate NaN values
    sensor_df = df.iloc[:, _SENSOR_COLS].copy()
    sensor_df = sensor_df.interpolate(method="linear", limit_direction="both")
    sensor_df = sensor_df.fillna(0.0)
    data = sensor_df.values.astype(np.float32)

    os.makedirs(_NPY_CACHE_DIR, exist_ok=True)
    np.save(cache_data,   data)
    np.save(cache_labels, labels)
    return data, labels


def _sliding_windows(data: np.ndarray, labels: np.ndarray,
                     win: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    T, C = data.shape
    X_wins, y_wins = [], []
    for start in range(0, T - win + 1, step):
        w_data   = data[start:start + win]
        w_labels = labels[start:start + win]
        counts   = np.bincount(w_labels, minlength=13)
        majority = int(np.argmax(counts))
        if majority == 0:
            continue
        X_wins.append(w_data.T)       # (C, win)
        y_wins.append(majority - 1)   # 1-12 → 0-11
    if not X_wins:
        return np.empty((0, C, win), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.stack(X_wins).astype(np.float32), np.array(y_wins, dtype=np.int32)


def _load_subjects(subject_ids: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, ss = [], [], []
    for sid in subject_ids:
        data, labels = _load_subject(sid)
        X_w, y_w     = _sliding_windows(data, labels, WINDOW_SIZE, WINDOW_STEP)
        if len(X_w) == 0:
            continue
        Xs.append(X_w)
        ys.append(y_w)
        ss.append(np.full(len(X_w), sid, dtype=np.int32))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss))


def _get_data():
    if "all" not in _CACHE:
        print(f"  [PAMAP2] Loading from {DATA_DIR} ...")
        X_tr, y_tr, s_tr = _load_subjects(_TRAIN_SUBJECTS)
        X_te, y_te, s_te = _load_subjects(_TEST_SUBJECTS)
        _CACHE["all"] = (X_tr, y_tr, s_tr, X_te, y_te, s_te)
        print(f"  [PAMAP2] Train: {len(X_tr)} windows | "
              f"Test: {len(X_te)} windows | "
              f"Channels: {NUM_CHANNELS} | Window: {WINDOW_SIZE} samples")
    return _CACHE["all"]


def compute_and_save_norm_stats() -> None:
    """Compute global normalization stats from all training subjects and save."""
    print(f"  [PAMAP2] Computing global norm stats from subjects {_TRAIN_SUBJECTS}...")
    X_tr, _, _, _, _, _ = _get_data()
    mu = X_tr.mean(axis=(0, 2), keepdims=True)
    sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    np.savez(_NORM_STATS_FILE, mu=mu, sd=sd)
    print(f"  [PAMAP2] Saved → {_NORM_STATS_FILE}  (mu shape={mu.shape})")


# ── Class balancing (ROS+RUS) ─────────────────────────────────

def _balance_ros_rus(X: np.ndarray, y: np.ndarray, beta: float = 1.0):
    primary  = np.argmax(y, axis=1)
    pc_counts = np.array([np.sum(primary == c) for c in range(NUM_CLASSES)])
    active   = pc_counts[pc_counts > 0]

    if len(active) == 0:
        return X, y

    m_l, m_s = int(active.max()), int(active.min())
    target   = int(m_s + (m_l - m_s) * beta)

    print(f"  [ROS+RUS] m_l={m_l}, m_s={m_s}, beta={beta}, target={target}")
    print(f"  [ROS+RUS] Before: {dict(zip(SC_NAMES, pc_counts))}")

    bX, by = [], []
    for c in range(NUM_CLASSES):
        idx = np.where(primary == c)[0]
        if len(idx) == 0:
            continue
        if len(idx) < target:
            idx = np.random.choice(idx, target, replace=True)
        elif len(idx) > target:
            idx = np.random.choice(idx, target, replace=False)
        bX.append(X[idx])
        by.append(y[idx])

    bX   = np.concatenate(bX)
    by   = np.concatenate(by)
    perm = np.random.permutation(len(bX))

    print(f"  [ROS+RUS] After:  {dict(zip(SC_NAMES, by[perm].sum(0).astype(int)))}")
    return bX[perm], by[perm]


# ── Data augmentation ─────────────────────────────────────────

def _augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.8:
        x = x + torch.randn_like(x) * 0.05
    if torch.rand(1).item() < 0.5:
        scale = 0.8 + 0.4 * torch.rand(x.size(0), x.size(1), 1, device=x.device)
        x = x * scale
    if torch.rand(1).item() < 0.5:
        shift = torch.randint(-10, 11, (1,)).item()
        if shift > 0:
            x = F.pad(x[:, :, shift:], (0, shift))
        elif shift < 0:
            x = F.pad(x[:, :, :shift], (-shift, 0))
    return x


# ── DataLoader construction ───────────────────────────────────

def load_data(partition_id: int, num_partitions: int, beta: float = 1.0,
              batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Build train/test DataLoaders for one FL client partition."""
    if num_partitions <= 0:
        raise ValueError(f"num_partitions must be > 0, got {num_partitions}")
    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(
            f"partition_id must be in [0, {num_partitions - 1}], got {partition_id}"
        )

    train_subjects = sorted(_TRAIN_SUBJECTS)
    subject_split = None
    if num_partitions <= len(train_subjects):
        chunks      = np.array_split(train_subjects, num_partitions)
        my_subjects = set(int(s) for s in chunks[partition_id])
    else:
        sid         = train_subjects[partition_id % len(train_subjects)]
        my_subjects = {int(sid)}
        same_pids   = [p for p in range(num_partitions)
                       if train_subjects[p % len(train_subjects)] == sid]
        if len(same_pids) > 1:
            subject_split = (same_pids.index(partition_id), len(same_pids))

    use_global_norm = os.getenv("FL_GLOBAL_NORM", "1") == "1"
    part_key = (tuple(sorted(my_subjects)), subject_split)

    if part_key not in _PART_CACHE:
        if use_global_norm and os.path.exists(_NORM_STATS_FILE):
            print(f"  [PAMAP2] Loading from {DATA_DIR} ...")
            X_part, y_part, _ = _load_subjects(list(my_subjects))
            X_te, y_te, _     = _load_subjects(_TEST_SUBJECTS)
            nstats = np.load(_NORM_STATS_FILE)
            mu = nstats['mu']
            sd = nstats['sd']
            print(f"  [PAMAP2] Partition {partition_id}: "
                  f"subjects={sorted(my_subjects)}, windows={len(X_part)}")
            print(f"  [PAMAP2] Normalization: global (pre-saved stats)")
        else:
            X_tr, y_tr, s_tr, X_te, y_te, _ = _get_data()
            my_idx  = np.where(np.isin(s_tr, list(my_subjects)))[0]
            X_part  = X_tr[my_idx].copy()
            y_part  = y_tr[my_idx].copy()
            if use_global_norm:
                mu = X_tr.mean(axis=(0, 2), keepdims=True)
                sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
            else:
                mu = X_part.mean(axis=(0, 2), keepdims=True)
                sd = X_part.std(axis=(0, 2), keepdims=True) + 1e-8
            print(f"  [PAMAP2] Partition {partition_id}: "
                  f"subjects={sorted(my_subjects)}, windows={len(X_part)}")
            print(f"  [PAMAP2] Normalization: "
                  f"{'global-train' if use_global_norm else 'partition-local'}")

        if len(X_part) == 0:
            raise ValueError(
                f"No training windows assigned to partition {partition_id} "
                f"(subjects={sorted(my_subjects)})."
            )

        if subject_split is not None:
            split_idx, n_splits = subject_split
            n = len(X_part)
            chunk = n // n_splits
            start = split_idx * chunk
            end = start + chunk if split_idx < n_splits - 1 else n
            X_part = X_part[start:end]
            y_part = y_part[start:end]
            print(f"  [PAMAP2] Partition {partition_id}: non-overlapping slice "
                  f"[{start}:{end}] ({end-start} windows, "
                  f"split {split_idx+1}/{n_splits} of subject {list(my_subjects)[0]})")

        _PART_CACHE[part_key] = (X_part, y_part, X_te, y_te, mu, sd)

    X_part, y_part, X_te, y_te, mu, sd = _PART_CACHE[part_key]

    X_part_n = (X_part - mu) / sd
    X_te_n   = (X_te   - mu) / sd

    y_part_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_part]
    y_te_oh   = np.eye(NUM_CLASSES, dtype=np.float32)[y_te]

    if beta > 0:
        X_part_n, y_part_oh = _balance_ros_rus(X_part_n, y_part_oh, beta=beta)

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

    trainloader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
    )
    testloader = DataLoader(
        TensorDataset(X_te_t, y_te_t),
        batch_size=batch_size,
    )
    return trainloader, testloader
