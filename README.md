# FLBL2: Real-World Layer-2 Blockchain Auditing for Federated Learning on Edge Hardware

Federated learning across 10 Raspberry Pi 4 devices with every training round committed on-chain to Base mainnet (Ethereum L2) via a Solidity smart contract and archived via IPFS.

Paper: [FLBL2 — Future Generation Computer Systems (submitted)](paper_elsevier/elsarticle-template-num.tex) | Contract: [`0xBE3FeAC76293711C5D32426860303AfE24cdf527`](https://basescan.org/address/0xBE3FeAC76293711C5D32426860303AfE24cdf527) on Base mainnet

---

## What this is

FLBL2 trains a 12-class human activity recognition model on the MHEALTH dataset, distributed across 10 physical Raspberry Pi 4 clients using [Flower](https://flower.ai). After each aggregation round the server writes three on-chain records (local, vote, global) and pins four artefacts to IPFS via Pinata. Any change to a stored model or to the on-chain record sequence is detectable via `verifyChain()`.

The repo includes two middleware variants — **baseline** (blocking IPFS, per-TX gas estimation) and **optimised** (async IPFS, cached gas parameters, deferred verification) — and the complete output logs from 20 real sessions run on Base mainnet in April 2026.

**Key numbers from the 200-round experiment:**

| Metric | Value |
|--------|-------|
| Sessions | 20 (10 baseline + 10 optimised) |
| Rounds | 200 (10 per session) |
| On-chain transactions | 620 |
| IPFS pins | 840 |
| Cost per round | < $0.012 (Base L2) |
| Peak accuracy (baseline) | 98.6% ± 1.6% |
| Peak accuracy (optimised) | 97.2% ± 2.1% |
| `verifyChain()` violations | 0 / 20 sessions |

---

## Repo layout

```
fl_blockchain_evm/
├── client_app.py          # Flower ClientApp (train + evaluate)
├── server_app.py          # Flower ServerApp (aggregation, blockchain writes)
├── task.py                # Flower task helpers
├── utils.py               # Shared utilities
├── core/
│   ├── model.py           # SE-ResNet (~860K params, 23-ch input, 12 classes)
│   ├── data.py            # MHEALTH loader, windowing, subject partitioning
│   ├── training.py        # train() / evaluate()
│   └── constants.py       # Activity labels
├── infra/
│   ├── blockchain.py      # web3.py wrapper — baseline and optimised modes
│   └── ipfs_storage.py    # Pinata upload / fetch
├── strategy/
│   └── medical_fedavg.py  # Equal-weight FedAvg aggregation
└── dashboard/
    ├── server.py           # FastAPI backend (REST + SSE)
    ├── fl_dashboard.html   # Live dashboard
    └── state.py            # Thread-safe state

contracts/
├── FLBL2.sol              # Solidity smart contract
└── FLBL2_abi.json         # ABI (copy from Remix after deployment)

data/
└── MHEALTHDATASET/        # Download separately — see Setup

outputs/                   # Created at runtime; published runs already included
├── baseline_YYYYMMDD_HHMMSS/
│   ├── results.json           # JSONL — one record per round event
│   ├── experiment_config.json
│   ├── perf_baseline_*.log    # Per-event timing log
│   └── cm_round_N.png
├── optimized_YYYYMMDD_HHMMSS/
│   └── ...
└── tx_pin_map.csv             # 840-row audit map: TX hash, CID, pin status per event

pyproject.toml             # Flower config and hyperparameters
.env                       # Secrets — never commit
```

---

## Setup

### Requirements

- Python 3.11
- Flower 1.23, PyTorch 2.3, web3.py 7.x, Pinata SDK 3.x
- Real Base mainnet ETH (roughly 0.001 ETH per 10-round session)
- A Pinata account with a JWT token (free tier is sufficient)

```bash
git clone https://github.com/shinratttensei1/FLBL2.git
cd FLBL2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt .
```

### Dataset

Download MHEALTH from the [UCI repository](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) and place the ten `.log` files under `data/MHEALTHDATASET/`. No preprocessing needed — windowing and normalisation run at load time.

**Split used in the paper:** subjects 1–8 across 10 training devices (some subjects shared across two devices), subjects 9–10 as a fixed held-out test set (514 windows, evaluated after each round).

### Smart contract

Deploy `contracts/FLBL2.sol` from [Remix IDE](https://remix.ethereum.org):

1. Paste the contract, compile with Solidity 0.8.20.
2. In **Deploy & Run Transactions** set the environment to **Injected Provider – MetaMask**, with MetaMask on Base mainnet (chain ID 8453, RPC `https://mainnet.base.org`).
3. Click **Deploy**. Copy the contract address and ABI into your `.env` and `contracts/FLBL2_abi.json`.

The published experiment uses contract `0xBE3FeAC76293711C5D32426860303AfE24cdf527` — you can read all 620 transactions on [Basescan](https://basescan.org/address/0xBE3FeAC76293711C5D32426860303AfE24cdf527).

### Environment variables

Create `.env` in the project root:

```env
BASE_RPC_URL=https://mainnet.base.org
PRIVATE_KEY=0x<YOUR_WALLET_PRIVATE_KEY>
CONTRACT_ADDRESS=0x<DEPLOYED_CONTRACT_ADDRESS>

IPFS_BACKEND=pinata
PINATA_JWT=<YOUR_PINATA_JWT_TOKEN>

# 0 = baseline, 1 = optimised
BLOCKCHAIN_OPTIMIZED=0
```

---

## Running

### Hyperparameters

All FL hyperparameters are in `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 10
fraction-train    = 1.0
local-epochs      = 5
lr                = 0.002
batch-size        = 64
num-partitions    = 8
```

### Start training

```bash
flwr run .
```

Each round writes exactly 3 transactions to Base mainnet. With `BLOCKCHAIN_OPTIMIZED=0` IPFS uploads block the round critical path; with `BLOCKCHAIN_OPTIMIZED=1` they run in a background thread.

### Live dashboard (optional)

```bash
python -m fl_blockchain_evm.dashboard.server
# then open http://localhost:8000
```

Shows live accuracy, F1, per-client losses, blockchain ledger, and IPFS pins.

### Verify chain integrity

```python
from fl_blockchain_evm.infra.blockchain import EVMBlockchain
bc = EVMBlockchain()
print(bc.verify_chain())   # True if no tampering detected
```

---

## Outputs

```
outputs/<run_folder>/
├── results.json            # JSONL — one object per round event (type: local/vote/global)
├── experiment_config.json
├── perf_<mode>_<ts>.log    # Per-operation timing: IPFS, estimateGas, gasPrice, TX, confirm
└── cm_round_N.png          # Confusion matrix at round N

outputs/tx_pin_map.csv      # CID, TX hash, pin status for all 840 pin events
```

**Reading results.json** (JSONL, not a JSON array):

```python
import json
with open("outputs/latest/results.json") as f:
    records = [json.loads(line) for line in f if line.strip()]
global_rounds = [r for r in records if r["type"] == "global"]
for r in global_rounds:
    print(r["round"], r["accuracy"], r["f1_macro"])
```

---

## Baseline vs. optimised modes

The `BLOCKCHAIN_OPTIMIZED` flag switches between two middleware behaviours:

| | Baseline | Optimised |
|--|----------|-----------|
| **O1** Gas estimation | `eth_estimateGas` called per TX (3×/round) | Cached once per session |
| **O2** Gas price | `eth_gasPrice` called per TX (3×/round) | Fetched once per round |
| **O3** IPFS uploads | Blocking on round critical path | Background thread, joins after last round |
| **O4** Chain verify | After every round — O(N²) total | Once at session end — O(N) |

Both modes write the same 3 transactions per round and produce the same on-chain data. Only timing differs.

**Savings measured over 310 baseline rounds:**

| Optimisation | Mean saving |
|---|---|
| O1 Gas limit cache | ~1,428 ms/round |
| O2 Gas price cache | ~350 ms/round |
| O3 Async IPFS | ~34,308 ms/round (variance eliminated) |
| O4 Deferred verify | ~534 ms/round |

---

## Model

SE-ResNet for 1-D time-series, ~860K trainable parameters.

```
Input: (B, 23, 256)  — 23 sensor channels × 256 samples at 50 Hz
  Stage 1: Conv1d(23→32,  k=7) → BN → ReLU → MaxPool → 2× SEResBlock(32)
  Stage 2: Conv1d(32→64,  k=5) → BN → ReLU → MaxPool → 2× SEResBlock(64)
  Stage 3: Conv1d(64→128, k=3) → BN → ReLU → MaxPool → 2× SEResBlock(128)
  Stage 4: Conv1d(128→256,k=3) → BN → ReLU → MaxPool → 1× SEResBlock(256)
  GlobalAvgPool → Dropout(0.3) → Linear(256→12)
```

Loss: Focal Loss (γ=2). Optimiser: AdamW (lr=0.002, wd=1e-4) with cosine-annealing. Augmentation: Mixup (α=0.3), Gaussian noise, amplitude jitter.

Gzip-compressed model checkpoint: **3,439,144 bytes** (bit-identical across all 200 rounds).

---

## IPFS artefacts

Four artefacts are pinned per training round:

| Artefact | Type | Approximate size |
|----------|------|-----------------|
| Local aggregate metrics | `LOCAL` | ~5 KB |
| Vote decisions | `VOTE` | ~3 KB |
| Global model weights (gzip) | `GLOBAL` | ~3.3 MB |
| Global evaluation metrics | `GLOBAL` | ~2 KB |

Only the 32-byte SHA-256 digest (raw multihash, stripping the 2-byte `0x1220` prefix) is stored on-chain. The full artefact lives on IPFS; the on-chain CID commitment makes any post-hoc modification detectable.

---

## Citation

If you use this code or data, please cite the accompanying paper:

```
Bayan, T., Mukhambetiyar, B., & Yazici, A. (2026). FLBL2: Real-World Layer-2 Blockchain
Auditing for Federated Learning on Edge Hardware. Future Generation Computer Systems.
```
