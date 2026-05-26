"""gather_pins.py — Export all Pinata pins to outputs/pinata_pins.csv.

Run after every pipeline to get a fresh, correctly labelled table:
    python gather_pins.py

Boundary detection : round_10_votes marks the end of each FL run.
Variant assignment : matched to run directory timestamps (server is UTC+5).
Excluded flag      : pins from runs in excluded_blocks.json are flagged.
"""

import csv, glob, json, os, requests
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

UTC_OFFSET = timedelta(hours=5)   # server local time = UTC+5

# ── Fetch all pins from both Pinata accounts ───────────────────────────────────
def fetch_all_pins(jwt: str, label: str) -> list:
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {jwt}"
    pins, offset, page = [], 0, 0
    while True:
        resp = session.get(
            "https://api.pinata.cloud/data/pinList",
            params={"status": "pinned", "pageLimit": 1000, "pageOffset": offset},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("rows", [])
        for r in rows:
            meta = r.get("metadata") or {}
            kv   = meta.get("keyvalues") or {}
            pins.append({
                "account":     label,
                "ipfs_hash":   r.get("ipfs_pin_hash", ""),
                "name":        meta.get("name", ""),
                "size_bytes":  r.get("size", ""),
                "date_pinned": r.get("date_pinned", ""),
                "mime_type":   r.get("mime_type", ""),
            })
        page += 1
        print(f"  [{label}] page {page}: {len(rows)} pins (total so far: {len(pins)})")
        if len(rows) < 1000:
            break
        offset += 1000
    return pins


# ── Build run directory timeline ───────────────────────────────────────────────
def build_run_timeline() -> list:
    """Return list of (utc_start, utc_end, variant, pinata_account, dir_name)."""
    runs = []
    for run_dir in sorted(
        glob.glob("outputs_ucihar/baseline_*") + glob.glob("outputs_ucihar/optimized_*")
    ):
        name  = os.path.basename(run_dir)
        parts = name.split("_")
        if len(parts) < 3:
            continue
        variant = parts[0]
        try:
            local_dt = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
        except ValueError:
            print(f"  [WARN] Could not parse timestamp from directory: {name}")
            continue
        utc_start = local_dt - UTC_OFFSET

        # Read pinata_account from experiment_config.json if available
        pinata_account = "unknown"
        cfg_path = os.path.join(run_dir, "experiment_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                pinata_account = str(cfg.get("pinata_account", "unknown"))
            except Exception:
                pass

        utc_end = utc_start
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
        runs.append((utc_start, utc_end, variant, pinata_account, name))
    runs.sort()
    return runs


def match_run(first_pin_utc: datetime, runs: list) -> tuple:
    """Return (variant, pinata_account) for the run that most recently started
    before first_pin_utc. Allows up to 15 min of clock skew / early async uploads.
    """
    best_start = None
    matched_variant, matched_account = "unknown", "unknown"
    for (utc_start, utc_end, variant, pinata_account, dname) in runs:
        if utc_start <= first_pin_utc + timedelta(minutes=15):
            if best_start is None or utc_start > best_start:
                best_start       = utc_start
                matched_variant  = variant
                matched_account  = pinata_account
    return matched_variant, matched_account


# ── Load excluded time windows from excluded_blocks.json ─────────────────────
def build_excluded_windows() -> list:
    """Return list of (start_dt, end_dt) for runs that must be excluded."""
    path = "outputs_ucihar/excluded_blocks.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    windows = []
    for grp in data.get("excluded_run_groups", []):
        blocks = [b for b in data["blocks"]
                  if grp["first_block"] <= b["block_index"] <= grp["last_block"]]
        if not blocks:
            continue
        t0 = datetime.fromtimestamp(blocks[0]["timestamp"])
        t1 = datetime.fromtimestamp(blocks[-1]["timestamp"])
        # Tight window: no pre-buffer (avoids catching tail of previous valid run),
        # 1-min post-buffer for any last async uploads
        windows.append((t0, t1 + timedelta(minutes=1)))
    return windows


def is_excluded(date_str: str, windows: list) -> bool:
    try:
        t = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        # Convert UTC pin time to local for comparison with blockchain timestamps
        t_local = t + UTC_OFFSET
        for (w0, w1) in windows:
            if w0 <= t_local <= w1:
                return True
    except Exception:
        pass
    return False


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    all_pins = []
    for jwt, label in [
        (os.getenv("PINATA_JWT",   ""), "account_1"),
        (os.getenv("PINATA_JWT_2", ""), "account_2"),
    ]:
        if not jwt:
            print(f"  [{label}] JWT not configured — skipping")
            continue
        print(f"Fetching {label} ...")
        all_pins.extend(fetch_all_pins(jwt, label))

    if not all_pins:
        print("No pins fetched — check PINATA_JWT environment variables.")
        return

    # Sort chronologically
    all_pins.sort(key=lambda p: p["date_pinned"])
    print(f"\nTotal pins fetched: {len(all_pins)}")

    # Detect run boundaries: round_10_votes = end of each run
    run_num = 1
    for pin in all_pins:
        pin["run_number"] = run_num
        if pin["name"].replace(".gz", "") == "round_10_votes":
            run_num += 1
    total_runs = max(p["run_number"] for p in all_pins)
    print(f"Detected runs     : {total_runs} (via round_10_votes boundary)")

    # Group by detected run
    by_run = defaultdict(list)
    for p in all_pins:
        by_run[p["run_number"]].append(p)

    # Match each run to a directory to get variant + pinata_account
    runs = build_run_timeline()
    print(f"Run directories   : {len(runs)}")

    run_variant = {}
    run_account = {}
    for rn, pins in sorted(by_run.items()):
        first_utc = datetime.fromisoformat(
            pins[0]["date_pinned"].replace("Z", "+00:00")
        ).replace(tzinfo=None)
        var, acct = match_run(first_utc, runs)
        run_variant[rn] = var
        run_account[rn] = acct

    # Overrides for runs whose directories no longer exist or whose first pin
    # falls in a gap (e.g. deleted runs created a large time gap). Only applies
    # when timestamp matching returns "unknown".
    OVERRIDES = {
        # run_number: (variant, pinata_account)
        # Add entries here if needed after inspecting summary output
    }
    for rn, (var, acct) in OVERRIDES.items():
        if run_variant.get(rn) == "unknown":
            run_variant[rn] = var
            run_account[rn] = acct
            print(f"  [override] run {rn} → {var} / account_{acct}")

    for pin in all_pins:
        rn = pin["run_number"]
        pin["variant"]        = run_variant.get(rn, "unknown")
        pin["pinata_account"] = run_account.get(rn, "unknown")

    # Flag excluded runs: if >50% of a detected run's pins fall inside an
    # excluded time window, the entire run is marked excluded. This avoids
    # false positives from boundary pins of adjacent valid runs.
    excl_windows = build_excluded_windows()
    excluded_run_numbers: set = set()
    for rn, pins in by_run.items():
        in_window = sum(1 for p in pins if is_excluded(p["date_pinned"], excl_windows))
        if in_window > len(pins) * 0.5:
            excluded_run_numbers.add(rn)

    for pin in all_pins:
        pin["excluded"] = "yes" if pin["run_number"] in excluded_run_numbers else "no"

    excl_count = sum(1 for p in all_pins if p["excluded"] == "yes")
    if excl_count:
        print(f"Excluded pins     : {excl_count} pins from runs: {sorted(excluded_run_numbers)}")

    # Add complete flag: a run is complete if it has ≥40 pins and is not excluded
    for pin in all_pins:
        rn    = pin["run_number"]
        cnt   = len(by_run[rn])
        excl  = pin["excluded"] == "yes"
        pin["complete"] = "yes" if cnt >= 40 and not excl else "no"

    # Write CSV
    os.makedirs("outputs_ucihar", exist_ok=True)
    out = "outputs_ucihar/pinata_pins.csv"
    fields = ["run_number", "variant", "pinata_account", "account",
              "ipfs_hash", "name", "size_bytes", "date_pinned",
              "mime_type", "excluded", "complete"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_pins)

    # Summary
    print(f"\n{'Run':>4}  {'Variant':10}  {'Pinata':8}  {'Acct':10}  {'Pins':>5}  {'Excl':>4}  {'Status'}")
    print("-" * 72)
    for rn in sorted(by_run):
        pins  = by_run[rn]
        var   = pins[0]["variant"]
        pacct = pins[0]["pinata_account"]
        acct  = pins[0]["account"]
        cnt   = len(pins)
        excl  = sum(1 for p in pins if p["excluded"] == "yes")
        ok    = cnt >= 40 and excl == 0
        status = "✓" if ok else ("⚠ EXCLUDED" if excl > 0 else "⚠ INCOMPLETE")
        print(f"{rn:>4}  {var:10}  {pacct:8}  {acct:10}  {cnt:>5}  {excl:>4}  {status}")

    baseline_ok  = sum(1 for rn in by_run
                       if by_run[rn][0]["variant"] == "baseline"
                       and len(by_run[rn]) >= 40
                       and all(p["excluded"] == "no" for p in by_run[rn]))
    optimized_ok = sum(1 for rn in by_run
                       if by_run[rn][0]["variant"] == "optimized"
                       and len(by_run[rn]) >= 40
                       and all(p["excluded"] == "no" for p in by_run[rn]))

    print(f"\nClean baseline runs  : {baseline_ok}/10")
    print(f"Clean optimized runs : {optimized_ok}/10")
    print(f"Written              : {out}")


if __name__ == "__main__":
    main()
