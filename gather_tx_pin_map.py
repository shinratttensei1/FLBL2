"""gather_tx_pin_map.py — Map every blockchain transaction to its Pinata pin.

Produces outputs/tx_pin_map.csv with one row per expected IPFS upload,
showing whether the pin succeeded or failed and linking to Basescan.

Run after training completes:
    python gather_tx_pin_map.py

Requires:
    - outputs/<run>/fl_server.log   (tx hashes)
    - outputs/<run>/results.json    (round timestamps)
    - contracts/FLBlockchain_abi.json
    - .env  (BASE_SEPOLIA_RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS,
              PINATA_JWT, PINATA_JWT_2)
"""

import csv, json, os, glob, re, requests
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

UTC_OFFSET = timedelta(hours=5)   # server local time = UTC+5
BASESCAN   = "https://basescan.org/tx"

# ── 1. Connect to contract ────────────────────────────────────────────────────
w3 = Web3(Web3.HTTPProvider(os.getenv("BASE_SEPOLIA_RPC_URL")))
with open("contracts/FLBL2_abi.json") as f:
    abi = json.load(f)
contract = w3.eth.contract(
    address=Web3.to_checksum_address(os.getenv("CONTRACT_ADDRESS")), abi=abi)


# ── 2. Fetch all Pinata pins ──────────────────────────────────────────────────
def fetch_pins(jwt: str, label: str) -> list:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {jwt}"
    resp = s.get("https://api.pinata.cloud/data/pinList",
                 params={"status": "pinned", "pageLimit": 1000}, timeout=30)
    resp.raise_for_status()
    return [{"account":     label,
             "ipfs_hash":   r["ipfs_pin_hash"],
             "name":        (r.get("metadata") or {}).get("name", ""),
             "date_pinned": r["date_pinned"]}
            for r in resp.json().get("rows", [])]


all_pins = []
for jwt, label in [(os.getenv("PINATA_JWT", ""), "account_1"),
                   (os.getenv("PINATA_JWT_2", ""), "account_2")]:
    if jwt:
        print(f"Fetching {label} ...")
        all_pins.extend(fetch_pins(jwt, label))
all_pins.sort(key=lambda p: p["date_pinned"])
print(f"Total pins: {len(all_pins)}")

# Name-based lookup
by_name: dict = defaultdict(list)
for pin in all_pins:
    by_name[pin["name"].replace(".gz", "")].append(pin)


# ── 3. Build run time windows ─────────────────────────────────────────────────
runs = []   # (utc_start, utc_end, name, variant, pinata_account)
for run_dir in sorted(glob.glob("outputs_ucihar/baseline_*") + glob.glob("outputs_ucihar/optimized_*"),
                      key=lambda d: os.path.basename(d).split("_", 1)[1]):
    name    = os.path.basename(run_dir)
    parts   = name.split("_")
    variant = parts[0]
    local_dt  = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
    utc_start = local_dt - UTC_OFFSET
    utc_end   = utc_start

    cfg_path = os.path.join(run_dir, "experiment_config.json")
    pinata_acct = "?"
    if os.path.exists(cfg_path):
        pinata_acct = str(json.load(open(cfg_path)).get("pinata_account", "?"))

    res_path = os.path.join(run_dir, "results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            for line in f:
                try:
                    o = json.loads(line.strip())
                    if isinstance(o, dict) and o.get("type") == "global":
                        t = datetime.fromisoformat(o["timestamp"]) - UTC_OFFSET
                        if t > utc_end:
                            utc_end = t
                except Exception:
                    pass
    runs.append((utc_start, utc_end, name, variant, pinata_acct))

run_idx = {r[2]: i for i, r in enumerate(runs)}


# ── 4. Parse tx hashes from fl_server.log files ───────────────────────────────
def parse_txs(run_dir: str, run_name: str, variant: str) -> list:
    log_path = os.path.join(run_dir, "fl_server.log")
    if not os.path.exists(log_path):
        return []
    txs = []
    current_round, current_type = None, None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"ROUND (\d+)", line)
            if m:
                current_round = int(m.group(1))
            for bt in ("LOCAL", "VOTE", "GLOBAL"):
                if f"Firing {bt} block" in line:
                    current_type = bt
            m = re.search(r"Transaction sent: ([0-9a-f]{64})", line)
            if m and current_round is not None:
                txs.append({
                    "run":        run_name,
                    "variant":    variant,
                    "fl_round":   current_round,
                    "block_type": current_type,
                    "tx_hash":    "0x" + m.group(1),
                    "basescan":   f"{BASESCAN}/0x{m.group(1)}",
                })

    # For baseline GLOBAL txs: decode embedded IPFS CID from calldata
    if variant == "baseline":
        for tx in txs:
            if tx["block_type"] == "GLOBAL":
                try:
                    raw = w3.eth.get_transaction(tx["tx_hash"])["input"]
                    decoded = contract.decode_function_input(raw)
                    payload = json.loads(
                        decoded[1].get("data", b"").decode("utf-8", errors="ignore")
                    )
                    tx["ipfs_model_cid"]   = payload.get("ipfs_model_cid", "")
                    tx["ipfs_metrics_cid"] = payload.get("ipfs_metrics_cid", "")
                except Exception:
                    tx["ipfs_model_cid"]   = ""
                    tx["ipfs_metrics_cid"] = ""
    return txs


all_txs = []
for run_info in runs:
    utc_start, utc_end, name, variant, pinata_acct = run_info
    run_dir = f"outputs_ucihar/{name}"
    print(f"Parsing {name} ...")
    all_txs.extend(parse_txs(run_dir, name, variant))
print(f"Total transactions: {len(all_txs)}")


# ── 5. Match each tx to its expected Pinata pin(s) ───────────────────────────
EXPECTED_PINS = {
    "LOCAL":  lambda r: [f"round_{r}_local"],
    "VOTE":   lambda r: [f"round_{r}_votes"],
    "GLOBAL": lambda r: [f"round_{r}_global_model", f"round_{r}_global_metrics"],
}

rows = []
for tx in all_txs:
    run_name = tx["run"]
    variant  = tx["variant"]
    fl_round = tx["fl_round"]
    btype    = tx["block_type"]
    if btype not in EXPECTED_PINS:
        continue

    run_info = next((r for r in runs if r[2] == run_name), None)
    if not run_info:
        continue
    utc_start, utc_end, _, _, pinata_acct = run_info

    # Extend the window to catch async uploads that finish during the next round
    idx = run_idx[run_name]
    next_run = runs[idx + 1] if idx + 1 < len(runs) else None
    late_limit = (next_run[0] if next_run else utc_end) + timedelta(minutes=15)

    for pin_stem in EXPECTED_PINS[btype](fl_round):
        pin_info = None

        # Baseline GLOBAL: try CID embedded in tx calldata first
        if variant == "baseline" and btype == "GLOBAL":
            cid_key = "ipfs_model_cid" if "model" in pin_stem else "ipfs_metrics_cid"
            cid = tx.get(cid_key, "")
            if cid:
                pin_info = next((p for p in all_pins if p["ipfs_hash"] == cid), None)

        # Fallback: match by pin name within time window
        if not pin_info:
            for p in by_name.get(pin_stem, []):
                t = datetime.fromisoformat(
                    p["date_pinned"].replace("Z", "+00:00")
                ).replace(tzinfo=None)
                if utc_start - timedelta(minutes=5) <= t <= late_limit:
                    pin_info = p
                    break

        rows.append({
            "run":                   run_name,
            "variant":               variant,
            "pinata_account":        pinata_acct,
            "fl_round":              fl_round,
            "block_type":            btype,
            "expected_pin":          pin_stem,
            "pin_status":            "success" if pin_info else "failed",
            "tx_hash":               tx["tx_hash"],
            "basescan_url":          tx["basescan"],
            "ipfs_hash":             pin_info["ipfs_hash"] if pin_info else "",
            "pinata_account_actual": pin_info["account"] if pin_info else "",
            "date_pinned":           pin_info["date_pinned"][:19] if pin_info else "",
        })

# Sort by actual execution order (timestamp in run directory name)
rows.sort(key=lambda r: r["run"].split("_", 1)[1])


# ── 6. Write CSV ──────────────────────────────────────────────────────────────
os.makedirs("outputs_ucihar", exist_ok=True)
out = "outputs_ucihar/tx_pin_map.csv"
fields = ["run", "variant", "pinata_account", "fl_round", "block_type",
          "expected_pin", "pin_status", "tx_hash", "basescan_url",
          "ipfs_hash", "pinata_account_actual", "date_pinned"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)


# ── 7. Summary ────────────────────────────────────────────────────────────────
total   = len(rows)
success = sum(1 for r in rows if r["pin_status"] == "success")
failed  = total - success
print(f"\nTotal : {total}  |  Success : {success}  |  Failed : {failed}")

fail_rows = [r for r in rows if r["pin_status"] == "failed"]
if fail_rows:
    by_run = defaultdict(list)
    for r in fail_rows:
        by_run[r["run"]].append(r["expected_pin"])
    print("\nFailed by run:")
    for run in sorted(by_run, key=lambda r: r.split("_", 1)[1]):
        print(f"  {run:45} {len(by_run[run]):>2}: {by_run[run]}")

print(f"\nWritten → {out}")
