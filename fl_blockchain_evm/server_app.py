import json
import os
import time as _time
from datetime import datetime
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

matplotlib.use('Agg')

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp

from fl_blockchain_evm.dashboard import state as live_state
from fl_blockchain_evm.infra.blockchain import EVMBlockchain as FLBlockchain
from fl_blockchain_evm.strategy.medical_fedavg import MedicalFedAvg
from fl_blockchain_evm.task import Net, test as test_fn, load_data, SC_NAMES, NUM_CLASSES
from fl_blockchain_evm.core.constants import DATASET_NAME
from fl_blockchain_evm.utils import get_device, print_table

G, Y, C, R = '\033[92m', '\033[93m', '\033[96m', '\033[0m'

# ── Per-run output directory ──────────────────────────────────────────────────
# Set EXPERIMENT_VARIANT=baseline or EXPERIMENT_VARIANT=optimized in the shell
# (or .env) before launching.  Each run gets its own timestamped subfolder so
# results never collide:
#   outputs/baseline_20260422_143022/
#   outputs/optimized_20260422_153011/
_VARIANT  = os.getenv("EXPERIMENT_VARIANT", "experiment")
_RUN_TS   = datetime.now().strftime("%Y%m%d_%H%M%S")
_OUT_BASE = os.getenv("OUTPUT_BASE_DIR", "outputs_pamap2")
_OUT_DIR  = f"{_OUT_BASE}/{_VARIANT}_{_RUN_TS}"
os.makedirs(_OUT_DIR, exist_ok=True)
os.makedirs(_OUT_BASE, exist_ok=True)

# Keep outputs/latest as a convenience symlink to the most-recent run
_LATEST = "outputs/latest"
try:
    if os.path.islink(_LATEST):
        os.unlink(_LATEST)
    os.symlink(os.path.abspath(_OUT_DIR), _LATEST)
except Exception:
    pass  # non-fatal on Windows or restricted filesystems

# Tee all stdout/stderr to outputs/<run>/fl_server.log so every print is persisted
import sys as _sys
import io as _io

class _Tee(_io.TextIOBase):
    """Write to both the original stream and a log file simultaneously."""
    def __init__(self, original, logpath):
        self._orig = original
        self._log  = open(logpath, "a", buffering=1, encoding="utf-8")
    def write(self, s):
        self._orig.write(s)
        self._orig.flush()
        try:
            self._log.write(s)
            self._log.flush()
        except Exception:
            pass
        return len(s)
    def flush(self):
        self._orig.flush()

_LOG_PATH = f"{_OUT_DIR}/fl_server.log"
_sys.stdout = _Tee(_sys.stdout, _LOG_PATH)
_sys.stderr = _Tee(_sys.stderr, _LOG_PATH)

# ── Experiment config snapshot ────────────────────────────────────────────────
# Written once at startup so every run folder is self-contained.
# Collect pyproject.toml config values (best-effort).
_pyproject_cfg: Dict = {}
try:
    import tomllib as _tomllib
    with open("pyproject.toml", "rb") as _f:
        _pt = _tomllib.load(_f)
    _pyproject_cfg = (_pt.get("tool", {}).get("flwr", {})
                         .get("app", {}).get("config", {}))
except Exception:
    pass

_PINATA_ACCOUNT = os.getenv("PINATA_ACCOUNT", "1")

_EXPERIMENT_CONFIG: Dict = {
    "variant":              _VARIANT,
    "run_timestamp":        _RUN_TS,
    "output_dir":           _OUT_DIR,
    "blockchain_optimized": os.getenv("BLOCKCHAIN_OPTIMIZED", "0") == "1",
    "ipfs_backend":         os.getenv("IPFS_BACKEND", "disabled"),
    "pinata_account":       _PINATA_ACCOUNT,
    "hardware_note":        os.getenv("HARDWARE_NOTE", "8x RPi4, WiFi 2.4GHz"),
    "num_rounds":           int(_pyproject_cfg.get("num-server-rounds", 10)),
    "num_clients":          int(_pyproject_cfg.get("num-partitions", os.getenv("NUM_PARTITIONS", 8))),
    "local_epochs":         int(_pyproject_cfg.get("local-epochs", 1)),
    "batch_size":           int(_pyproject_cfg.get("batch-size", 256)),
    "lr":                   float(_pyproject_cfg.get("lr", 0.002)),
    "started_at":           datetime.now().isoformat(),
}
with open(f"{_OUT_DIR}/experiment_config.json", "w") as _f:
    json.dump(_EXPERIMENT_CONFIG, _f, indent=2)
print(f"[SETUP] Run output dir : {_OUT_DIR}")
print(f"[SETUP] Variant        : {_VARIANT}")
print(f"[SETUP] Pinata account : {_PINATA_ACCOUNT}")

# Blockchain is only initialized on server, not on clients
# This prevents crashes when blockchain credentials are missing
_blockchain = None

def _init_blockchain():
    """Initialize blockchain only once, on first use."""
    global _blockchain
    if _blockchain is None:
        try:
            _blockchain = FLBlockchain(output_dir=_OUT_DIR)
        except (ValueError, ConnectionError) as e:
            print(f"\n{Y}[WARNING] Blockchain unavailable: {e}{R}")
            print(f"{Y}Continuing without blockchain recording...{R}\n")
            _blockchain = None
    return _blockchain

_round_state: Dict = {
    "train_results":  [],
    "current_round":  0,
    "loss_mean":      0.0,
    "loss_std":       0.0,
    "threshold":      0.0,
    "round_start_t":  0.0,   # wall-clock time when training results arrived
}

_EVAL_NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS", "7"))


def _plot_cm(cm, rnd, acc, f1):
    if isinstance(cm, list):
        cm = np.array(cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=SC_NAMES, yticklabels=SC_NAMES)
    plt.title(f'Confusion Matrix — Round {rnd}\nAcc: {acc:.2%} | F1: {f1:.4f}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f"{_OUT_DIR}/cm_round_{rnd}.png", dpi=150, bbox_inches='tight')
    plt.close()


def global_evaluate(server_round, arrays, config=None):
    live_state.evaluating(server_round)

    model = Net()
    dev = get_device()
    model.load_state_dict(arrays.to_torch_state_dict())
    model.to(dev)

    # ── Model payload size (bytes sent to each client, FP32 weights) ──────
    model_payload_bytes = sum(p.numel() * 4 for p in model.parameters())

    _, testloader = load_data(0, _EVAL_NUM_PARTITIONS, beta=0)
    m = test_fn(model, testloader, dev)

    print(f"\n{Y}{'═'*60}{R}")
    print(f"{Y}  [ROUND {server_round}] GLOBAL — {NUM_CLASSES} Activities ({DATASET_NAME}){R}")
    print(f"{Y}{'═'*60}{R}")
    for k in ["loss", "accuracy", "f1_macro", "f1_weighted",
              "precision_macro", "recall_macro", "specificity_macro", "auc_macro"]:
        print(f"   {k:20s}: {m[k]:.4f}")
    print(f"   {'── Per-Activity ──':20s}")
    for i, sc in enumerate(SC_NAMES):
        print(f"     {sc:5s}  P={m['per_class_precision'][i]:.3f}  "
              f"R={m['per_class_recall'][i]:.3f}  "
              f"F1={m['per_class_f1'][i]:.3f}  "
              f"AUC={m['per_class_auc'][i]:.3f}  "
              f"(n={m['per_class_support'][i]})")

    try:
        _plot_cm(m["confusion_matrix"], server_round, m["accuracy"], m["f1_macro"])
    except Exception as _e:
        print(f"  [WARN] confusion matrix plot failed (non-fatal): {_e}")

    num_clients = len(_round_state["train_results"])

    bc = _init_blockchain()
    if bc is not None:
        bc.add_global_model_block(
            fl_round=server_round,
            model_state_dict=arrays.to_torch_state_dict(),
            accuracy=m["accuracy"],
            f1_macro=m["f1_macro"],
            auc_macro=m["auc_macro"],
            loss=m["loss"],
            num_clients=num_clients,
        )

        chain_length = bc.get_chain_length()
        chain_valid = bc.verify_chain()

        summary = bc.get_round_summary(server_round)
        print(f"\n{C}  [BLOCKCHAIN] Round {server_round} complete — "
              f"3 blocks written (LOCAL + VOTE + GLOBAL) | "
              f"Chain length: {summary['total_blocks']}{R}")
    else:
        chain_length = 0
        chain_valid = None
        print(f"\n{Y}  [BLOCKCHAIN] Skipped (not configured){R}")

    _round_state["train_results"] = []

    # ── Live state update ──
    live_state.round_complete(server_round, m, chain_length, chain_valid)

    # ── Include IPFS CIDs if available ──
    round_cids = None
    if bc is not None:
        round_cids = bc.get_round_cids(server_round)
        if round_cids:
            live_state.ipfs_pinned(server_round, round_cids)

    log = {k: m[k] for k in [
        "loss", "accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro", "specificity_macro", "auc_macro",
        "per_class_f1", "per_class_precision", "per_class_recall",
        "per_class_auc", "per_class_support", "confusion_matrix",
        "num_samples", "num_classes",
    ]}
    round_wall_time_s = round(_time.time() - _round_state["round_start_t"], 2)
    log.update({
        "round":               server_round,
        "type":                "global",
        "timestamp":           datetime.now().isoformat(),
        "superclass_names":    SC_NAMES,
        "optimal_thresholds":  m.get("optimal_thresholds", [0.5] * NUM_CLASSES),
        "blockchain_blocks":   chain_length if bc is not None else 0,
        "round_wall_time_s":   round_wall_time_s,
        "model_payload_bytes": model_payload_bytes,
        "variant":             _VARIANT,
    })
    # Include IPFS CIDs if available (already handled above)
    if round_cids:
        log["ipfs_cids"] = round_cids
    print(f"  [TIMING] Round wall time: {round_wall_time_s:.1f}s | "
          f"Model payload: {model_payload_bytes/1024:.0f} KB")
    with open(f"{_OUT_DIR}/results.json", "a") as f:
        json.dump(log, f)
        f.write("\n")

    return {
        "loss":      m["loss"],
        "accuracy":  m["accuracy"],
        "f1_macro":  m["f1_macro"],
        "auc_macro": m["auc_macro"],
    }


_rnd = {"train": 0, "eval": 0}


def train_metrics_aggregation(metrics_list, weighting_key):
    _rnd["train"] += 1
    rnd = _rnd["train"]
    _round_state["current_round"] = rnd

    _round_state["round_start_t"] = _time.time()
    live_state.round_started(rnd)

    data = []
    for m in sorted(metrics_list,
                    key=lambda x: int(x["metrics"].get("client_id", 0))):
        met = m["metrics"]
        data.append({
            "client_id":      int(met.get("client_id", 0)),
            "train_loss":     float(met.get("train_loss", 0)),
            "num_examples":   int(met.get("num-examples", 0)),
            "training_time":  float(met.get("training_time_seconds", 0)),
            "active_classes": int(met.get("active_classes", 0)),
            "cpu_percent":    float(met.get("cpu_percent", -1.0)),
            "ram_used_mb":    float(met.get("ram_used_mb", -1.0)),
            "cpu_temp_c":     float(met.get("cpu_temp_c", -1.0)),
        })

    _round_state["train_results"] = data

    print(f"\n{C}  ROUND {rnd} TRAINING: {len(data)} devices{R}")
    print_table(
        ["Device", "Loss", "Samples", "Time(s)", "Cls"],
        [[d["client_id"], f"{d['train_loss']:.4f}", d["num_examples"],
          f"{d['training_time']:.1f}", d["active_classes"]] for d in data],
        ["Device", "Loss", "Samples", "Time(s)", "Cls"],
    )

    losses = [d["train_loss"] for d in data]
    loss_mean = float(np.mean(losses))
    loss_std = float(np.std(losses)) if len(losses) > 1 else 0.0
    threshold = loss_mean + loss_std

    _round_state["loss_mean"] = loss_mean
    _round_state["loss_std"] = loss_std
    _round_state["threshold"] = threshold

    print(f"  Loss  mean={loss_mean:.4f}  std={loss_std:.4f}  "
          f"threshold={threshold:.4f}")

    bc = _init_blockchain()
    if bc is not None:
        print(f"\n{C}  [BLOCKCHAIN] Firing LOCAL + VOTE blocks for Round {rnd} "
              f"(async)...{R}")

        votes = bc.add_round_summary_block(
            fl_round=rnd,
            clients=data,
            loss_mean=loss_mean,
            loss_std=loss_std,
            threshold=threshold,
        )
    else:
        print(f"\n{Y}  [BLOCKCHAIN] Skipped (not configured){R}")
        votes = [{
            "client_id": d["client_id"],
            "loss": d["train_loss"],
            "vote": "ACCEPT",
        } for d in data]

    print_table(
        ["Device", "Loss", "Verdict"],
        [[v["client_id"], f"{v['loss']:.4f}", v["vote"]] for v in votes],
        ["Device", "Loss", "Verdict"],
    )

    # ── Live state update ──
    live_state.clients_trained(data, loss_mean, loss_std, threshold, votes)

    with open(f"{_OUT_DIR}/results.json", "a") as f:
        json.dump({
            "round":      rnd,
            "type":       "device_training",
            "timestamp":  datetime.now().isoformat(),
            "loss_mean":  loss_mean,
            "loss_std":   loss_std,
            "threshold":  threshold,
            "devices":    data,
        }, f)
        f.write("\n")

    return MetricRecord({
        "train_loss_avg": float(np.mean(losses)),
        "num_devices":    float(len(data)),
    })


def weighted_average(metrics_list, weighting_key):
    _rnd["eval"] += 1
    rnd = _rnd["eval"]
    total = sum(int(m["metrics"]["num-examples"]) for m in metrics_list)
    if total == 0:
        return MetricRecord({"eval_acc": 0.0, "eval_f1": 0.0})

    def wavg(k):
        return sum(
            float(m["metrics"][k]) * int(m["metrics"]["num-examples"])
            for m in metrics_list
        ) / total

    data = []
    for m in sorted(metrics_list,
                    key=lambda x: int(x["metrics"]["client_id"])):
        met = m["metrics"]
        d = {k: float(met[k]) if k != "client_id" else int(met[k])
             for k in ["client_id", "eval_loss", "eval_acc",
                       "eval_f1", "eval_auc", "num-examples"]}
        # Include per-client eval time and device stats if reported
        for _opt_key in ("eval_time_seconds", "cpu_percent", "ram_used_mb", "cpu_temp_c"):
            if _opt_key in met:
                d[_opt_key] = float(met[_opt_key])
        data.append(d)

    print(f"\n{G}  ROUND {rnd} EVALUATION: {len(data)} devices{R}")
    print_table(
        ["Device", "Loss", "Acc", "F1", "AUC", "N"],
        [[d["client_id"], f"{d['eval_loss']:.4f}", f"{d['eval_acc']:.4f}",
          f"{d['eval_f1']:.4f}", f"{d['eval_auc']:.4f}", int(d["num-examples"])]
         for d in data],
        ["Device", "Loss", "Acc", "F1", "AUC", "N"],
    )

    with open(f"{_OUT_DIR}/results.json", "a") as f:
        json.dump([{
            "type":      "client_eval",
            "round":     rnd,
            "timestamp": datetime.now().isoformat(),
            **d,
        } for d in data], f)
        f.write("\n")

    return MetricRecord({
        k: wavg(k) for k in [
            "eval_acc", "eval_f1", "eval_f1_weighted",
            "eval_precision", "eval_recall",
            "eval_specificity", "eval_auc",
        ]
    })


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):
    global _EVAL_NUM_PARTITIONS, _VARIANT, _OUT_DIR

    # Resolve experiment variant from run_config (env var not visible inside FAB subprocess)
    run_variant = str(context.run_config.get("experiment-variant", _VARIANT))
    if run_variant != _VARIANT:
        new_out_dir = f"{_OUT_BASE}/{run_variant}_{_RUN_TS}"
        try:
            if os.path.exists(_OUT_DIR) and not os.path.exists(new_out_dir):
                os.rename(_OUT_DIR, new_out_dir)
            else:
                os.makedirs(new_out_dir, exist_ok=True)
            _VARIANT = run_variant
            _OUT_DIR = new_out_dir
            if os.path.islink(_LATEST):
                os.unlink(_LATEST)
            os.symlink(os.path.abspath(_OUT_DIR), _LATEST)
            _EXPERIMENT_CONFIG["variant"] = _VARIANT
            with open(f"{_OUT_DIR}/experiment_config.json", "w") as _f:
                json.dump(_EXPERIMENT_CONFIG, _f, indent=2)
        except Exception as _e:
            print(f"[SETUP] Warning: could not update output dir for variant {run_variant}: {_e}")

    # run_config from pyproject.toml; env vars as fallback for flower-server-app direct use
    lr         = float(context.run_config.get("lr",               os.getenv("LR",           "0.002")))
    num_rounds = int(  context.run_config.get("num-server-rounds", os.getenv("NUM_ROUNDS",   "10")))
    frac       = float(context.run_config.get("fraction-train",    os.getenv("FRACTION_TRAIN","1.0")))
    _EVAL_NUM_PARTITIONS = int(context.run_config.get(
        "num-partitions", os.getenv("NUM_PARTITIONS", str(_EVAL_NUM_PARTITIONS))
    ))

    # Push blockchain-optimized into os.environ so EVMBlockchain.__init__ picks it up
    bc_opt = int(context.run_config.get("blockchain-optimized", 0))
    os.environ['BLOCKCHAIN_OPTIMIZED'] = str(bc_opt)

    if os.path.exists(f"{_OUT_DIR}/results.json"):
        os.remove(f"{_OUT_DIR}/results.json")

    # ── Init blockchain ──
    bc = _init_blockchain()
    contract_address = bc.contract_address if bc is not None else "not-configured"

    # ── Init live state ──
    live_state.init(num_rounds, contract_address)

    model = Net()
    strategy = MedicalFedAvg(
        fraction_train=frac,
        train_metrics_aggr_fn=train_metrics_aggregation,
        evaluate_metrics_aggr_fn=weighted_average,
    )

    ipfs_status = "enabled" if (bc is not None and bc.ipfs_enabled) else "disabled"
    print(f"\n{C}{'═'*60}")
    print(f"  {NUM_CLASSES} Activities ({DATASET_NAME}): {', '.join(SC_NAMES)}")
    print(f"  Rounds: {num_rounds} | LR: {lr} | Device: {get_device()}")
    print(f"  Eval partitioning: {_EVAL_NUM_PARTITIONS} partitions")
    print(f"  Blockchain: 3 tx per round (LOCAL + VOTE + GLOBAL)")
    print(f"  IPFS:       {ipfs_status}")
    print(f"  Dashboard:  open dashboard.html in your browser")
    print(f"{'═'*60}{R}\n")

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    bc = _init_blockchain()
    chain_len = bc.get_chain_length() if bc is not None else 0
    live_state.done(chain_len)
    if bc is not None:
        bc.print_chain_summary()

    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    print(f"\n{G}   Done. Model  -> final_model.pt")
    print(f"    Metrics -> {_OUT_DIR}/results.json{R}")
