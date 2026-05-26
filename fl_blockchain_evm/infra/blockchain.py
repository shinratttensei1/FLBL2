"""Simplified EVM Blockchain wrapper for FLBlockchain contract.
Features:
  - _send_transaction supports fire-and-wait pattern: returns tx_hash immediately
    without blocking; call wait_for_pending() at end of round to confirm all.
  - add_round_summary_block: writes ONE LOCAL block and ONE VOTE block summarising
    ALL clients for the round, instead of one pair of blocks per client.
    This reduces blockchain writes from (2K + 1) to 3 per round.
  - OPTIMIZED mode (BLOCKCHAIN_OPTIMIZED=1 in .env):
      * Lazy gas-limit cache: estimate_gas called once per block-type (round 0),
        result cached with 25% headroom and reused for rounds 1-N — zero RPC
        calls for gas estimation from round 1 onwards
      * Round-level gas_price caching (one RPC fetch per round, not per tx)
      * Async IPFS upload via background thread (off critical path)
      * verifyChain() deferred to end of session only
  - PerfLogger writes per-round timing to a timestamped .log file so
    baseline and optimized runs can be compared directly.
"""

import os
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional
from web3 import Web3
from dotenv import load_dotenv

# ── Performance logger ───────────────────────────────────────

def _make_perf_logger(optimized: bool, output_dir: str = "outputs") -> logging.Logger:
    """Create a file logger for per-round timing comparison."""
    os.makedirs(output_dir, exist_ok=True)
    tag = "optimized" if optimized else "baseline"
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{output_dir}/perf_{tag}_{ts}.log"

    logger = logging.getLogger(f"fl_perf_{tag}_{ts}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S.%f")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"=== FL-Blockchain Performance Log ===")
    logger.info(f"Mode     : {'OPTIMIZED' if optimized else 'BASELINE'}")
    logger.info(f"Started  : {datetime.now().isoformat()}")
    logger.info(f"File     : {path}")
    logger.info("=" * 50)
    print(f"  [PERF] Logging to {path}")
    return logger

# Baseline fallback gas limit (wei) — used only before the lazy cache is
# populated on the first estimate.  Generous enough to avoid OOG; unused gas
# is refunded by the EVM so over-estimating is safe.
_GAS_FALLBACK = 500_000

# Try to load .env from multiple possible locations
_env_loaded = False
_current_dir = os.getcwd()

# Get the directory of this script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))  # Go up two levels to project root

# Try loading from project root
_env_path = os.path.join(_project_root, '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path)
    _env_loaded = True

# Also try current directory and parent directories
if not _env_loaded:
    for _env_path in ['.env', '../.env', '../../.env']:
        if os.path.exists(_env_path):
            load_dotenv(_env_path)
            _env_loaded = True
            break

# If no .env file found, try loading from environment variables directly
if not _env_loaded:
    # This is normal for production deployments where env vars are set externally
    pass


class EVMBlockchain:
    """Wrapper for FLBlockchain smart contract."""

    def __init__(self, output_dir: str = "outputs"):
        # ── Optimized mode flag ───────────────────────────────
        self._optimized = os.getenv("BLOCKCHAIN_OPTIMIZED", "0").strip() == "1"
        self._perf = _make_perf_logger(self._optimized, output_dir)
        self._perf.info(f"BLOCKCHAIN_OPTIMIZED = {self._optimized}")

        # Load config
        self.rpc_url = os.getenv("BASE_SEPOLIA_RPC_URL")
        self.private_key = os.getenv("PRIVATE_KEY")
        self.contract_address = os.getenv("CONTRACT_ADDRESS")

        missing_vars = []
        if not self.rpc_url:
            missing_vars.append("BASE_SEPOLIA_RPC_URL")
        if not self.private_key:
            missing_vars.append("PRIVATE_KEY")
        if not self.contract_address:
            missing_vars.append("CONTRACT_ADDRESS")

        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}. "
                           f"Make sure .env file exists in the project root with these variables set.")

        # Connect to Base Sepolia
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to {self.rpc_url}")

        print(
            f"  [EVM] Connected to network (Chain ID: {self.w3.eth.chain_id})")

        # Load account
        self.account = self.w3.eth.account.from_key(self.private_key)
        print(f"  [EVM] Using account: {self.account.address}")

        # Check balance
        balance = self.w3.eth.get_balance(self.account.address)
        balance_eth = self.w3.from_wei(balance, 'ether')
        print(f"  [EVM] Balance: {balance_eth:.4f} ETH")

        # Load contract
        with open('contracts/FLBL2_abi.json', 'r') as f:
            contract_abi = json.load(f)

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contract_address),
            abi=contract_abi
        )
        print(f"  [EVM] Contract loaded: {self.contract_address}")

        # Verify contract
        chain_length = self.contract.functions.getBlockCount().call()
        print(f"  [EVM] Current chain length: {chain_length} blocks")

        # Pending tx hashes queued for confirmation at end of round
        self._pending: List[bytes] = []
        # Timestamps matching _pending so we can log submission→confirmation gap
        self._pending_t0: List[float] = []

        # Nonce tracked manually so fire-and-wait works without collisions
        self._nonce: Optional[int] = None

        # Local chain length counter — keeps us from re-querying the chain
        # every round (each round writes exactly 3 blocks: LOCAL+VOTE+GLOBAL)
        self._chain_length: int = chain_length

        # Per-round cached gas price (OPTIMIZED mode: fetched once per round)
        self._cached_gas_price: Optional[int] = None

        # Per-block-type gas cache (OPTIMIZED mode: estimate_gas once, reuse).
        # Populated lazily on the first transaction of each block_type so that
        # the measured value is used; this avoids the brittleness of hard-coded
        # static limits while still skipping estimate_gas RPC calls from round 1
        # onwards.
        self._gas_cache: Dict[str, int] = {}

        # IPFS background threads (OPTIMIZED mode): joined after blockchain
        # confirmation so CIDs are captured before the round completes.
        # Each entry: (thread, fl_round, result_dict)
        # result_dict keys: local_cid | vote_cid | model_cid | metrics_cid
        self._ipfs_threads: List[tuple] = []

        # IPFS off-chain storage (optional — degrades gracefully)
        self._ipfs = None
        self._round_cids: Dict[int, Dict[str, str]] = {}
        try:
            ipfs_backend = os.getenv("IPFS_BACKEND", "").strip()
            if ipfs_backend:
                from fl_blockchain_evm.infra.ipfs_storage import IPFSStorage
                self._ipfs = IPFSStorage(backend=ipfs_backend)
                print(f"  [EVM] IPFS storage enabled ({ipfs_backend} backend)")
        except Exception as e:
            print(f"  [EVM] IPFS not available: {e}")

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _next_nonce(self) -> int:
        """Return next nonce, incrementing our local counter."""
        if self._nonce is None:
            self._nonce = self.w3.eth.get_transaction_count(
                self.account.address)
        n = self._nonce
        self._nonce += 1
        return n

    def _send_transaction(self, function_call, fire_and_wait: bool = False,
                          block_type: str = ""):
        """Send a transaction.

        fire_and_wait=False (default): send and block until confirmed.
        fire_and_wait=True           : send immediately, store hash in
                                       self._pending, return without blocking.
                                       Call wait_for_pending() when ready.

        BASELINE  — calls estimate_gas + fresh gas_price per transaction.
        OPTIMIZED — uses _STATIC_GAS map + cached round-level gas_price.
        """
        nonce = self._next_nonce()

        # ── Gas limit ─────────────────────────────────────────
        t_gas0 = time.perf_counter()
        if self._optimized and block_type and block_type in self._gas_cache:
            # O1: reuse the value measured on the first round — zero RPC calls.
            gas_limit = self._gas_cache[block_type]
            t_gas1 = time.perf_counter()
            self._perf.info(
                f"  gas_limit  type={block_type:6s} cached={gas_limit}"
                f"  saved_rpc_ms=~30"
            )
        else:
            # Baseline path, or first use of this block_type in optimized mode:
            # call estimate_gas (safe, accurate) and cache the result.
            try:
                gas_estimate = function_call.estimate_gas(
                    {'from': self.account.address})
                gas_limit = int(gas_estimate * 1.25)  # 25 % safety headroom
            except Exception:
                gas_limit = _GAS_FALLBACK
            t_gas1 = time.perf_counter()
            if self._optimized and block_type:
                self._gas_cache[block_type] = gas_limit   # cache for round 2+
                self._perf.info(
                    f"  gas_limit  type={block_type:6s} estimated={gas_limit}"
                    f"  estimate_ms={1000*(t_gas1-t_gas0):.1f}  (cached for future rounds)"
                )
            else:
                self._perf.info(
                    f"  gas_limit  type={block_type:6s} estimated={gas_limit}"
                    f"  estimate_ms={1000*(t_gas1-t_gas0):.1f}"
                )

        # ── Gas price ─────────────────────────────────────────
        t_gp0 = time.perf_counter()
        if self._optimized and self._cached_gas_price is not None:
            gas_price = self._cached_gas_price
            t_gp1 = time.perf_counter()
            self._perf.info(
                f"  gas_price  cached={gas_price}  saved_rpc_ms=~20"
            )
        else:
            gas_price = self.w3.eth.gas_price
            t_gp1 = time.perf_counter()
            self._perf.info(
                f"  gas_price  fetched={gas_price}"
                f"  fetch_ms={1000*(t_gp1-t_gp0):.1f}"
            )

        transaction = function_call.build_transaction({
            'from': self.account.address,
            'nonce': nonce,
            'gas': gas_limit,
            'gasPrice': gas_price,
            'chainId': self.w3.eth.chain_id,
        })

        signed = self.w3.eth.account.sign_transaction(
            transaction, self.private_key)

        t_send = time.perf_counter()
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        t_sent = time.perf_counter()

        self._perf.info(
            f"  tx_sent    type={block_type:6s}"
            f"  hash={tx_hash.hex()[:12]}..."
            f"  send_ms={1000*(t_sent-t_send):.1f}"
        )
        print(f"  [EVM] Transaction sent: {tx_hash.hex()}")

        if fire_and_wait:
            self._pending.append(tx_hash)
            self._pending_t0.append(t_sent)
            return tx_hash

        print(f"  [EVM] Waiting for confirmation...")
        receipt = self.w3.eth.wait_for_transaction_receipt(
            tx_hash, timeout=600, poll_latency=1.0)
        t_conf = time.perf_counter()
        if receipt['status'] == 1:
            self._perf.info(
                f"  confirmed  type={block_type:6s}"
                f"  block={receipt['blockNumber']}"
                f"  confirm_ms={1000*(t_conf-t_sent):.0f}"
            )
            print(f"  [EVM] ✓ Confirmed in block {receipt['blockNumber']}")
            return receipt
        else:
            raise RuntimeError(f"Transaction failed: {receipt}")

    def wait_for_pending(self, timeout: int = 600):
        """Block until all pending (fire-and-wait) transactions are confirmed."""
        if not self._pending:
            return
        n = len(self._pending)
        print(f"  [EVM] Waiting for {n} pending transaction(s)...")
        t_wait0 = time.perf_counter()
        for i, tx_hash in enumerate(self._pending):
            receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=timeout, poll_latency=1.0)
            if receipt['status'] != 1:
                raise RuntimeError(f"Transaction failed: {receipt}")
            t_conf = time.perf_counter()
            lag = 1000 * (t_conf - self._pending_t0[i]) if self._pending_t0 else 0
            self._perf.info(
                f"  confirmed  idx={i}"
                f"  block={receipt['blockNumber']}"
                f"  submit_to_confirm_ms={lag:.0f}"
            )
            print(f"  [EVM] ✓ Confirmed in block {receipt['blockNumber']}")
        t_wait1 = time.perf_counter()
        self._perf.info(
            f"  wait_for_pending  n={n}"
            f"  total_wait_ms={1000*(t_wait1-t_wait0):.0f}"
        )
        self._chain_length += n   # update local counter
        self._pending.clear()
        self._pending_t0.clear()

    # ─────────────────────────────────────────────────────────
    # Per-round batch writes  (3 tx per round instead of 2K+1)
    # ─────────────────────────────────────────────────────────

    def add_round_summary_block(
        self,
        fl_round: int,
        clients: List[Dict],
        loss_mean: float,
        loss_std: float,
        threshold: float,
    ):
        """Write ONE LOCAL block and ONE VOTE block summarising all clients.

        clients: list of dicts with keys
            client_id, train_loss, num_examples, training_time, active_classes

        Both transactions are fired immediately (fire_and_wait=True).
        Call wait_for_pending() after add_global_model_block() to confirm
        all three round transactions together.

        OPTIMIZED mode: gas_price is cached once here for the entire round
        (used for all 3 txs), and IPFS uploads run in background threads so
        the LOCAL/VOTE transactions are not blocked by HTTP upload latency.
        """
        t_round0 = time.perf_counter()
        self._perf.info(f"--- ROUND {fl_round} start  mode={'OPT' if self._optimized else 'BASE'}")

        # ── Cache gas price once per round (OPTIMIZED) ────────
        if self._optimized:
            t_gp = time.perf_counter()
            self._cached_gas_price = self.w3.eth.gas_price
            self._perf.info(
                f"  gas_price_cached  price={self._cached_gas_price}"
                f"  fetch_ms={1000*(time.perf_counter()-t_gp):.1f}"
            )
        else:
            self._cached_gas_price = None  # re-fetch per tx in baseline

        # ── LOCAL block ───────────────────────────────────────
        local_payload = {
            "round": fl_round,
            "num_clients": len(clients),
            "clients": [
                {
                    "client_id":    c["client_id"],
                    "train_loss":   c["train_loss"],
                    "num_examples": c["num_examples"],
                    "training_time_seconds": c["training_time"],
                    "active_classes": c["active_classes"],
                }
                for c in clients
            ],
            "loss_mean": loss_mean,
            "loss_std":  loss_std,
            "threshold": threshold,
        }

        local_cid = None
        if self._ipfs:
            if self._optimized:
                # Fire IPFS upload in background; CID will arrive before
                # wait_for_pending() is called (blockchain confirmation >> IPFS upload)
                _local_result: Dict = {}
                def _upload_local():
                    try:
                        _local_result["local_cid"] = self._ipfs.pin_json(
                            local_payload, f"round_{fl_round}_local")
                        self._perf.info(
                            f"  ipfs_local  cid={_local_result['local_cid'][:16]}... [background]"
                        )
                    except Exception as e:
                        self._perf.info(f"  ipfs_local  ERROR={e}")
                t_ipfs0 = time.perf_counter()
                _local_thread = threading.Thread(target=_upload_local, daemon=True)
                _local_thread.start()
                self._ipfs_threads.append((_local_thread, fl_round, _local_result))
                self._perf.info(
                    f"  ipfs_local  background_thread_started"
                    f"  t={1000*(time.perf_counter()-t_ipfs0):.1f}ms"
                )
            else:
                t_ipfs0 = time.perf_counter()
                try:
                    local_cid = self._ipfs.pin_json(
                        local_payload, f"round_{fl_round}_local")
                    local_payload["ipfs_cid"] = local_cid
                except Exception as e:
                    print(f"  [IPFS] Warning: LOCAL pin failed: {e}")
                self._perf.info(
                    f"  ipfs_local  cid={str(local_cid)[:16]}..."
                    f"  upload_ms={1000*(time.perf_counter()-t_ipfs0):.0f} [blocking]"
                )

        local_data = json.dumps(local_payload)

        print(
            f"\n  [EVM] Firing LOCAL block (Round {fl_round}, {len(clients)} clients)...")
        t_tx0 = time.perf_counter()
        self._send_transaction(
            self.contract.functions.addBlock(
                fl_round,
                "LOCAL",
                Web3.to_bytes(text=local_data),
            ),
            fire_and_wait=True,
            block_type="LOCAL",
        )
        self._perf.info(
            f"  local_tx_fired  ms={1000*(time.perf_counter()-t_tx0):.1f}"
        )

        # ── VOTE block ────────────────────────────────────────
        votes = [
            {
                "client_id": c["client_id"],
                "vote":      "ACCEPTED" if c["train_loss"] <= threshold else "REJECTED",
                "loss":      c["train_loss"],
                "reason":    (
                    "loss within threshold"
                    if c["train_loss"] <= threshold
                    else f"loss {c['train_loss']:.4f} > threshold {threshold:.4f}"
                ),
            }
            for c in clients
        ]

        accepted = sum(1 for v in votes if v["vote"] == "ACCEPTED")
        rejected = len(votes) - accepted

        vote_payload = {
            "round":    fl_round,
            "threshold": threshold,
            "accepted": accepted,
            "rejected": rejected,
            "votes":    votes,
        }

        vote_cid = None
        if self._ipfs:
            if self._optimized:
                _vote_result: Dict = {}
                def _upload_vote():
                    try:
                        _vote_result["vote_cid"] = self._ipfs.pin_json(
                            vote_payload, f"round_{fl_round}_votes")
                        self._perf.info(
                            f"  ipfs_vote   cid={_vote_result['vote_cid'][:16]}... [background]"
                        )
                    except Exception as e:
                        self._perf.info(f"  ipfs_vote   ERROR={e}")
                t_ipfs1 = time.perf_counter()
                _vote_thread = threading.Thread(target=_upload_vote, daemon=True)
                _vote_thread.start()
                self._ipfs_threads.append((_vote_thread, fl_round, _vote_result))
                self._perf.info(
                    f"  ipfs_vote   background_thread_started"
                    f"  t={1000*(time.perf_counter()-t_ipfs1):.1f}ms"
                )
            else:
                t_ipfs1 = time.perf_counter()
                try:
                    vote_cid = self._ipfs.pin_json(
                        vote_payload, f"round_{fl_round}_votes")
                    vote_payload["ipfs_cid"] = vote_cid
                except Exception as e:
                    print(f"  [IPFS] Warning: VOTE pin failed: {e}")
                self._perf.info(
                    f"  ipfs_vote   cid={str(vote_cid)[:16]}..."
                    f"  upload_ms={1000*(time.perf_counter()-t_ipfs1):.0f} [blocking]"
                )

        vote_data = json.dumps(vote_payload)

        print(f"  [EVM] Firing VOTE block (Round {fl_round}: "
              f"{accepted} accepted / {rejected} rejected)...")
        t_tx1 = time.perf_counter()
        self._send_transaction(
            self.contract.functions.addBlock(
                fl_round,
                "VOTE",
                Web3.to_bytes(text=vote_data),
            ),
            fire_and_wait=True,
            block_type="VOTE",
        )
        self._perf.info(
            f"  vote_tx_fired  ms={1000*(time.perf_counter()-t_tx1):.1f}"
        )

        # Track IPFS CIDs for this round
        if self._ipfs and (local_cid or vote_cid):
            self._round_cids.setdefault(fl_round, {})
            if local_cid:
                self._round_cids[fl_round]["local_cid"] = local_cid
            if vote_cid:
                self._round_cids[fl_round]["vote_cid"] = vote_cid

        return votes  # returned so server_app can print the table

    def add_global_model_block(
        self,
        fl_round: int,
        model_state_dict,
        accuracy: float,
        f1_macro: float,
        auc_macro: float,
        loss: float,
        num_clients: int,
    ):
        """Write GLOBAL block, then wait for all three round txs together.

        BASELINE:  pins global model weights to IPFS *before* firing the tx
                   (blocking); calls verifyChain() after confirmation.
        OPTIMIZED: pins model in a background thread; fires GLOBAL tx
                   immediately; verifyChain() is skipped per-round and
                   called only at session end via print_chain_summary().
        """
        global_payload = {
            "accuracy":    accuracy,
            "f1_macro":    f1_macro,
            "auc_macro":   auc_macro,
            "loss":        loss,
            "num_clients": num_clients,
        }

        model_cid   = None
        metrics_cid = None

        if self._ipfs:
            if self._optimized:
                # Background upload — model (~3.4 MB gzip) off the critical path
                _global_result: Dict = {}
                def _upload_global():
                    try:
                        if model_state_dict is not None:
                            _global_result["model_cid"] = self._ipfs.pin_model(
                                model_state_dict, f"round_{fl_round}_global_model")
                        _global_result["metrics_cid"] = self._ipfs.pin_json(
                            global_payload, f"round_{fl_round}_global_metrics")
                        self._perf.info(
                            f"  ipfs_global model_cid="
                            f"{str(_global_result.get('model_cid',''))[:16]}..."
                            f"  [background]"
                        )
                    except Exception as e:
                        self._perf.info(f"  ipfs_global ERROR={e}")
                t_ipfs2 = time.perf_counter()
                _global_thread = threading.Thread(target=_upload_global, daemon=True)
                _global_thread.start()
                self._ipfs_threads.append((_global_thread, fl_round, _global_result))
                self._perf.info(
                    f"  ipfs_global background_thread_started"
                    f"  t={1000*(time.perf_counter()-t_ipfs2):.1f}ms"
                )
            else:
                t_ipfs2 = time.perf_counter()
                try:
                    if model_state_dict is not None:
                        model_cid = self._ipfs.pin_model(
                            model_state_dict,
                            f"round_{fl_round}_global_model",
                        )
                        global_payload["ipfs_model_cid"] = model_cid
                    metrics_cid = self._ipfs.pin_json(
                        global_payload,
                        f"round_{fl_round}_global_metrics",
                    )
                    global_payload["ipfs_metrics_cid"] = metrics_cid
                except Exception as e:
                    print(f"  [IPFS] Warning: GLOBAL pin failed: {e}")
                self._perf.info(
                    f"  ipfs_global model_cid={str(model_cid)[:16]}..."
                    f"  upload_ms={1000*(time.perf_counter()-t_ipfs2):.0f} [blocking]"
                )

        data = json.dumps(global_payload)

        print(f"\n  [EVM] Firing GLOBAL block (Round {fl_round})...")
        t_tx2 = time.perf_counter()
        self._send_transaction(
            self.contract.functions.addBlock(
                fl_round,
                "GLOBAL",
                Web3.to_bytes(text=data),
            ),
            fire_and_wait=True,
            block_type="GLOBAL",
        )
        self._perf.info(
            f"  global_tx_fired  ms={1000*(time.perf_counter()-t_tx2):.1f}"
        )

        # Track IPFS CIDs
        if self._ipfs and (model_cid or metrics_cid):
            cids = self._round_cids.setdefault(fl_round, {})
            if model_cid:
                cids["model_cid"] = model_cid
            if metrics_cid:
                cids["metrics_cid"] = metrics_cid

        # Now wait for LOCAL + VOTE + GLOBAL together
        print(
            f"  [EVM] Waiting for all Round {fl_round} transactions to confirm...")
        t_conf0 = time.perf_counter()
        self.wait_for_pending()
        t_conf1 = time.perf_counter()
        self._perf.info(
            f"  round_confirmation_total_ms={1000*(t_conf1-t_conf0):.0f}"
        )

        # ── Collect IPFS CIDs from background threads (OPTIMIZED) ─────
        if self._optimized and self._ipfs_threads:
            for thread, rnd, result in self._ipfs_threads:
                thread.join(timeout=30)
                if thread.is_alive():
                    self._perf.warning(
                        "IPFS background thread did not finish within 30s "
                        "for round %d — CIDs may be missing", rnd
                    )
                    print(
                        f"  [IPFS] WARNING: background upload thread timed out"
                        f" for round {rnd}",
                        flush=True,
                    )
                else:
                    d = self._round_cids.setdefault(rnd, {})
                    for key in ("local_cid", "vote_cid", "model_cid", "metrics_cid"):
                        if key in result:
                            d[key] = result[key]
            self._ipfs_threads.clear()

        # ── verifyChain: per-round in baseline, deferred in optimized ─
        if not self._optimized:
            t_vc = time.perf_counter()
            valid = self.verify_chain()
            self._perf.info(
                f"  verify_chain  valid={valid}"
                f"  ms={1000*(time.perf_counter()-t_vc):.0f} [per-round]"
            )

        self._perf.info(
            f"--- ROUND {fl_round} end"
            f"  total_overhead_ms="  # rough: from first tx fired to confirmation
            f"{1000*(t_conf1-t_tx2):.0f}"
        )

    # ─────────────────────────────────────────────────────────
    # Query helpers
    # ─────────────────────────────────────────────────────────

    def verify_chain(self) -> bool:
        return self.contract.functions.verifyChain().call()

    def get_chain_length(self) -> int:
        """Return cached chain length (updated after every wait_for_pending call)."""
        return self._chain_length

    def get_round_summary(self, fl_round: int) -> Dict:
        return {
            "fl_round":     fl_round,
            "total_blocks": self._chain_length,
        }

    def print_chain_summary(self):
        length = self.get_chain_length()
        # In optimized mode verifyChain was deferred; run it now (once)
        t_vc = time.perf_counter()
        is_valid = self.verify_chain()
        self._perf.info(
            f"  verify_chain  valid={is_valid}"
            f"  ms={1000*(time.perf_counter()-t_vc):.0f}"
            f"  {'[deferred-end]' if self._optimized else '[per-round-final]'}"
        )
        print(f"\n  {'═'*60}")
        print(f"  EVM BLOCKCHAIN SUMMARY")
        print(f"  {'═'*60}")
        print(f"  Contract:        {self.contract_address}")
        print(f"  Total blocks:    {length}")
        print(f"  Chain integrity: {'✓ VALID' if is_valid else '✗ BROKEN'}")
        if self._ipfs:
            stats = self._ipfs.get_session_stats()
            print(f"  IPFS backend:    {stats['backend']}")
            print(f"  IPFS pins:       {stats['total_pins']}")
            print(
                f"  IPFS uploaded:   {stats['total_bytes_uploaded']:,} bytes")
        print(f"  {'═'*60}\n")

    # ─────────────────────────────────────────────────────────
    # IPFS helpers
    # ─────────────────────────────────────────────────────────

    @property
    def ipfs_enabled(self) -> bool:
        """True if IPFS storage backend is configured and available."""
        return self._ipfs is not None

    def get_round_cids(self, fl_round: int) -> Optional[Dict[str, str]]:
        """Return IPFS CIDs pinned for a given FL round, or None."""
        return self._round_cids.get(fl_round)

    def get_all_cids(self) -> Dict[int, Dict[str, str]]:
        """Return all IPFS CIDs indexed by round number."""
        return dict(self._round_cids)

    def get_ipfs_storage(self):
        """Return the underlying IPFSStorage instance (or None)."""
        return self._ipfs


FLBlockchain = EVMBlockchain
