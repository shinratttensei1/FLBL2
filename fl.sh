#!/usr/bin/env bash
# fl.sh — Federated Learning orchestrator for N Raspberry Pi 4s
#
#   ./fl.sh setup    One-time setup: deploy code + data, install deps
#   ./fl.sh train    Run federated learning with live dashboard
#   ./fl.sh stop     Stop all FL processes on laptop and Pis
#   ./fl.sh logs     Stream live logs from all devices
#   ./fl.sh status   Show training progress and system status
#
# ── TO ADD MORE PIs: just add their hostname to PI_HOSTS below ──
#
# Configuration via environment variables:
#   PI_USER=pi         SSH user on all Pis (default: pi)
#   NUM_ROUNDS=10      Training rounds (default: 10)
#   LOCAL_EPOCHS=1     Local training epochs per round (default: 1)
#   LR=0.002           Learning rate (default: 0.002)
#   BATCH_SIZE=256     Batch size (default: 256)
#   SERVER_IP=...      Override auto-detected laptop IP

set -euo pipefail

# ── Pi hostnames — ADD NEW PIs HERE ───────────────────────────
PI_HOSTS=(
    "raspberrypi1.local"   # partition 0
    "raspberrypi.local"    # partition 1
    "raspberrypi2.local"   # partition 2
    "raspberrypi3.local"   # partition 3
    "raspberrypi4.local"   # partition 4
    "raspberrypi5.local"   # partition 5
    "raspberrypi6.local"   # partition 6
    "raspberrypi7.local"   # partition 7
    "raspberrypi8.local"   # partition 8
    "raspberrypi9.local"   # partition 9
)

# ── Training defaults ──────────────────────────────────────────
PI_USER="${PI_USER:-pi}"
NUM_ROUNDS="${NUM_ROUNDS:-10}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_SESSIONS="${N_SESSIONS:-10}"

# Derived automatically — do not change
NUM_PARTITIONS="${#PI_HOSTS[@]}"

# ── Flower ports ───────────────────────────────────────────────
readonly SL_SERVERAPPIO_PORT=9091
readonly SL_FLEET_PORT=9092
readonly SL_CONTROL_PORT=9093
readonly SN_CLIENTAPPIO_PORT=9094
readonly DASHBOARD_PORT=8080

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_DIR="/tmp/fl_logs"

# ── Colours ────────────────────────────────────────────────────
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'
C='\033[0;36m'; B='\033[1m'; N='\033[0m'

ts()   { date '+%Y-%m-%d %H:%M:%S'; }
info() { echo -e "${G}[$(ts)] INFO ${N} $*"; }
warn() { echo -e "${Y}[$(ts)] WARN ${N} $*"; }
err()  { echo -e "${R}[$(ts)] ERROR${N} $*" >&2; }
step() { echo -e "\n${C}${B}══ $* ══${N}  [$(ts)]"; }
die()  { err "$*"; exit 1; }

pi_ssh() {
    local host="$1"; shift
    ssh -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        -o BatchMode=yes \
        -o ServerAliveInterval=30 \
        "${PI_USER}@${host}" "$@"
}

pi_scp() {
    scp -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        -o BatchMode=yes \
        "$@"
}

check_pi() { pi_ssh "$1" "echo ok" >/dev/null 2>&1; }

# Ping all Pis in parallel to wake their WiFi adapters from power-save doze,
# then SSH in and turn off power management for the duration of the session.
wake_pis() {
    local hosts=("$@")
    info "  Pinging Pis to wake WiFi adapters..."
    local pids=()
    for pi in "${hosts[@]}"; do
        ping -c 2 -W 2 "$pi" >/dev/null 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
    sleep 2

    info "  Disabling WiFi power management on all Pis (session)..."
    local pids2=()
    for pi in "${hosts[@]}"; do
        pi_ssh "$pi" "sudo iwconfig wlan0 power off 2>/dev/null || true" 2>/dev/null &
        pids2+=($!)
    done
    for p in "${pids2[@]}"; do wait "$p" 2>/dev/null || true; done
}

server_ip() {
    local ip=""
    for iface in en0 en1 en2 en3; do
        ip=$(ipconfig getifaddr "$iface" 2>/dev/null) && [ -n "$ip" ] && echo "$ip" && return
    done
    ip=$(ip route get 8.8.8.8 2>/dev/null | awk '/src/{print $7; exit}')
    [ -n "$ip" ] && echo "$ip" && return
    hostname -I 2>/dev/null | awk '{print $1}'
}

activate_venv() {
    cd "$PROJECT_DIR"
    [ -d "venv" ] && source venv/bin/activate || true
}

# ─────────────────────────────────────────────────────────────
#  SETUP — deploy code + data, install deps on all Pis
# ─────────────────────────────────────────────────────────────
cmd_setup() {
    step "FL SETUP — ${NUM_PARTITIONS} Pi(s)"

    info "Checking SSH connectivity..."
    for pi in "${PI_HOSTS[@]}"; do
        check_pi "$pi" \
            && info "  ✓ $pi" \
            || die "Cannot reach $pi.\n  Enable SSH: sudo raspi-config → Interface Options → SSH\n  Verify: ssh ${PI_USER}@${pi}"
    done

    step "Deploying to ${NUM_PARTITIONS} Pi(s) in parallel"
    mkdir -p "$LOG_DIR"

    _deploy() {
        local pi_host="$1"
        local logf="$LOG_DIR/setup_${pi_host}.log"
        {
            echo "[$(ts)] ── Setup: ${pi_host} ──"

            echo "[$(ts)] Syncing source code..."
            pi_ssh "$pi_host" "rm -rf ~/FL-Blockchain-EVM ~/FL-Blockchain-EVM-tmp" 2>/dev/null || true
            tar -czf - \
                --exclude='venv' --exclude='.git' --exclude='__pycache__' \
                --exclude='*.pyc' --exclude='outputs' --exclude='.env' \
                --exclude='final_model.pt' --exclude='*.bak' \
                -C "$(dirname "$PROJECT_DIR")" "$(basename "$PROJECT_DIR")" \
            | pi_ssh "$pi_host" \
                "cd ~ \
                 && tar -xzf - --warning=no-unknown-keyword 2>/dev/null \
                 && dirname_tar=\$(ls -d ~/FL-Blockchain-EVM* 2>/dev/null | head -1) \
                 && [ -d ~/FL-Blockchain-EVM ] || mv \"\$dirname_tar\" ~/FL-Blockchain-EVM \
                 && chmod +x ~/FL-Blockchain-EVM/fl.sh \
                 && echo '[$(ts)] Code synced'"

            echo "[$(ts)] Syncing MHEALTH data..."
            local data_src="$PROJECT_DIR/data/MHEALTHDATASET"
            pi_ssh "$pi_host" "mkdir -p ~/FL-Blockchain-EVM/data/MHEALTHDATASET/.npy_cache"
            # Copy real log files if present
            if ls "$data_src"/mHealth_subject*.log 1>/dev/null 2>&1; then
                pi_scp -q "$data_src"/mHealth_subject*.log \
                    "${PI_USER}@${pi_host}:~/FL-Blockchain-EVM/data/MHEALTHDATASET/"
                echo "[$(ts)] Copied $(ls "$data_src"/mHealth_subject*.log | wc -l | tr -d ' ') subject files."
            fi
            # Copy npy cache if present
            if ls "$data_src/.npy_cache"/s*.npy 1>/dev/null 2>&1; then
                pi_scp -q "$data_src/.npy_cache"/s*.npy \
                    "${PI_USER}@${pi_host}:~/FL-Blockchain-EVM/data/MHEALTHDATASET/.npy_cache/"
                echo "[$(ts)] Copied npy cache."
            fi

            [ -f "$PROJECT_DIR/.env" ] && \
                pi_scp -q "$PROJECT_DIR/.env" "${PI_USER}@${pi_host}:~/FL-Blockchain-EVM/.env" && \
                echo "[$(ts)] .env copied."

            echo "[$(ts)] Installing Python dependencies (may take 15-30 min first time)..."
            pi_ssh "$pi_host" bash << 'PISETUP'
set -e
cd ~/FL-Blockchain-EVM
_l() { echo "[$(date '+%H:%M:%S')] $*"; }

_l "Disabling WiFi power management (persistent)..."
sudo iwconfig wlan0 power off 2>/dev/null || true
sudo mkdir -p /etc/NetworkManager/conf.d
sudo tee /etc/NetworkManager/conf.d/wifi-pm.conf > /dev/null << 'WIFICFG'
[connection]
wifi.powersave = 2
WIFICFG
sudo tee /etc/network/if-up.d/disable-wifi-pm > /dev/null << 'IFUP'
#!/bin/sh
iwconfig wlan0 power off 2>/dev/null || true
IFUP
sudo chmod +x /etc/network/if-up.d/disable-wifi-pm 2>/dev/null || true
_l "WiFi power management disabled."

_l "apt-get update..."
sudo apt-get update -qq 2>/dev/null || true
sudo apt-get install -y -qq build-essential python3 python3-venv python3-dev \
    libopenblas-dev libblas-dev liblapack-dev git curl 2>/dev/null || true

_l "Creating venv..."
[ -d "venv" ] || python3 -m venv venv
source venv/bin/activate

_l "Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet

_l "Installing fl dependencies..."
pip install -r requirements-pi.txt \
    2>&1 | grep -E '(Successfully installed|error|ERROR|already satisfied)' | head -20 || true

_l "Verifying..."
python3 - << 'V'
import sys
for pkg in [("flwr","__version__"),("torch","__version__"),
            ("numpy","__version__"),("sklearn","__version__")]:
    try:
        m = __import__(pkg[0])
        print(f"  ✓ {pkg[0]:<12} {getattr(m, pkg[1], '?')}")
    except ImportError:
        print(f"  ✗ {pkg[0]:<12} MISSING")
        sys.exit(1)
V

which flower-supernode >/dev/null 2>&1 \
    && _l "  ✓ flower-supernode: $(which flower-supernode)" \
    || { _l "  ✗ flower-supernode not found"; exit 1; }

_l "Setup COMPLETE on $(hostname)"
PISETUP
            echo "[$(ts)] ✓ ${pi_host} done."
        } 2>&1 | tee "$logf"
    }

    # Launch all deployments in parallel
    local pids=()
    for pi in "${PI_HOSTS[@]}"; do
        _deploy "$pi" &
        pids+=($!)
    done

    local ok=true
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { err "${PI_HOSTS[$i]} failed → $LOG_DIR/setup_${PI_HOSTS[$i]}.log"; ok=false; }
    done
    [ "$ok" = "true" ] || exit 1

    step "SETUP COMPLETE"
    for i in "${!PI_HOSTS[@]}"; do
        info "  Pi $i (${PI_HOSTS[$i]}) — partition $i"
    done
    info "  Run './fl.sh train' to start training."
}

# ─────────────────────────────────────────────────────────────
#  TRAIN — full FL run: SuperLink + dashboard + SuperNodes + flwr run
# ─────────────────────────────────────────────────────────────
cmd_train() {
    step "FL TRAINING — $NUM_ROUNDS rounds, $NUM_PARTITIONS partitions"

    local SRV_IP="${SERVER_IP:-$(server_ip)}"
    [ -n "$SRV_IP" ] || die "Could not detect laptop IP.\nSet: SERVER_IP=192.168.x.x ./fl.sh train"

    info "  Server IP       : $SRV_IP"
    for i in "${!PI_HOSTS[@]}"; do
        info "  Pi $i (part. $i)  : ${PI_HOSTS[$i]}"
    done
    info "  Rounds          : $NUM_ROUNDS"
    info "  LR / Epochs     : $LR / $LOCAL_EPOCHS"
    info "  Flower ports    : Fleet=$SL_FLEET_PORT  Control=$SL_CONTROL_PORT"
    info "  Dashboard       : http://localhost:$DASHBOARD_PORT/monitor"

    wake_pis "${PI_HOSTS[@]}"

    for pi in "${PI_HOSTS[@]}"; do
        check_pi "$pi" || die "$pi unreachable — run './fl.sh setup' first"
        info "  ✓ $pi reachable"
    done

    activate_venv
    mkdir -p "$LOG_DIR" "outputs"
    rm -f "$LOG_DIR"/*.log outputs/results.json outputs/fl_server.log
    for f in training superlink dashboard; do : > "$LOG_DIR/${f}.log"; done
    : > "outputs/fl_server.log"

    # ── Cleanup on Ctrl+C or exit ────────────────────────────
    _SL_PID="" _DASH_PID=""
    cleanup() {
        step "SHUTDOWN"
        [ -n "$_SL_PID"   ] && kill "$_SL_PID"   2>/dev/null && info "  ✓ SuperLink stopped"   || true
        [ -n "$_DASH_PID" ] && kill "$_DASH_PID"  2>/dev/null && info "  ✓ Dashboard stopped"   || true
        pkill -f "flower-superlink" 2>/dev/null || true
        for pi in "${PI_HOSTS[@]}"; do
            pi_ssh "$pi" "pkill -f flower-supernode 2>/dev/null; true" 2>/dev/null &
        done
        wait; info "  ✓ Pi processes stopped"
    }
    trap cleanup EXIT

    # ── 1. Dashboard ─────────────────────────────────────────
    step "1/4  Dashboard"
    python run_dashboard.py > "$LOG_DIR/dashboard.log" 2>&1 &
    _DASH_PID=$!
    sleep 2
    kill -0 "$_DASH_PID" 2>/dev/null \
        && info "  ✓ Dashboard running → http://localhost:$DASHBOARD_PORT/monitor" \
        || { warn "  Dashboard failed (training continues)"; warn "  Log: $LOG_DIR/dashboard.log"; }

    # ── 2. SuperLink ─────────────────────────────────────────
    step "2/4  Flower SuperLink"
    pkill -f "flower-superlink" 2>/dev/null || true; sleep 1

    export FL_DATA_DIR="$PROJECT_DIR/data/MHEALTHDATASET"
    cd "$PROJECT_DIR"
    flower-superlink \
        --insecure \
        --serverappio-api-address "0.0.0.0:${SL_SERVERAPPIO_PORT}" \
        --fleet-api-address       "0.0.0.0:${SL_FLEET_PORT}" \
        --control-api-address     "0.0.0.0:${SL_CONTROL_PORT}" \
        > "$LOG_DIR/superlink.log" 2>&1 &
    _SL_PID=$!
    sleep 3

    kill -0 "$_SL_PID" 2>/dev/null \
        || { err "SuperLink failed to start"; cat "$LOG_DIR/superlink.log"; exit 1; }
    info "  ✓ SuperLink PID=$_SL_PID"
    info "    Fleet API    (supernodes) : $SRV_IP:$SL_FLEET_PORT"
    info "    Control API  (flwr run)   : $SRV_IP:$SL_CONTROL_PORT"

    # ── 3. SuperNodes on all Pis ─────────────────────────────
    step "3/4  SuperNodes on ${NUM_PARTITIONS} Pi(s)"

    _start_supernode() {
        local pi_host="$1"
        local part_id="$2"
        local logf="$LOG_DIR/supernode_${part_id}.log"

        local script
        script=$(cat << PIRUN
set -e
LOGF="/tmp/fl_client_${part_id}.log"
: > "\$LOGF"
_l() { echo "[\$(date '+%Y-%m-%d %H:%M:%S')] \$*" | tee -a "\$LOGF"; }

_l "════════════════════════════════════════════════════"
_l "  FL SuperNode — Partition ${part_id} / ${NUM_PARTITIONS}"
_l "  Host      : \$(hostname)  (${pi_host})"
_l "  Server    : ${SRV_IP}:${SL_FLEET_PORT}"
_l "  FL_DATA_DIR: \$HOME/FL-Blockchain-EVM/data/MHEALTHDATASET"
_l "════════════════════════════════════════════════════"

cd ~/FL-Blockchain-EVM
source venv/bin/activate

pkill -f flower-supernode 2>/dev/null && _l "Killed stale supernode" || true
sleep 1

export FL_DATA_DIR="\$HOME/FL-Blockchain-EVM/data/MHEALTHDATASET"

_l "Starting flower-supernode (detached from SSH)..."
nohup flower-supernode \\
    --insecure \\
    --superlink "${SRV_IP}:${SL_FLEET_PORT}" \\
    --node-config "partition-id=${part_id} num-partitions=${NUM_PARTITIONS}" \\
    --clientappio-api-address "0.0.0.0:${SN_CLIENTAPPIO_PORT}" \\
    >> "\$LOGF" 2>&1 &
disown \$!
_l "SuperNode running (PID=\$!), detached from SSH."
PIRUN
)
        echo "$script" | pi_ssh "$pi_host" bash > "$logf" 2>&1 &
        sleep 2
        info "  ✓ SuperNode ${part_id} started on ${pi_host}"
        info "    Remote log : ssh ${PI_USER}@${pi_host} tail -f /tmp/fl_client_${part_id}.log"
        info "    Local copy : $logf"
    }

    for i in "${!PI_HOSTS[@]}"; do
        _start_supernode "${PI_HOSTS[$i]}" "$i"
    done

    info ""
    info "  Waiting 15s for SuperNodes to register with SuperLink..."
    sleep 15

    local connected
    connected=$(grep -Eic "New node|registered|activate|pullmessages|supernode" "$LOG_DIR/superlink.log" 2>/dev/null || true)
    connected="${connected:-0}"
    [ "$connected" -ge 1 ] \
        && info "  ✓ SuperLink reports activity (connections detected)" \
        || warn "  No connection events yet in SuperLink log — SuperNodes may still be connecting"

    # ── 4. Run training ──────────────────────────────────────
    step "4/4  FL Training  (flwr run)"

    if ! grep -q '^\[tool\.flwr\.federations\]' pyproject.toml; then
        cat >> pyproject.toml << 'EOF'

[tool.flwr.federations]
default = "remote-federation"
EOF
        info "  pyproject.toml → added [tool.flwr.federations]"
    fi

    if ! grep -q '^\[tool\.flwr\.federations\.remote-federation\]' pyproject.toml; then
        cat >> pyproject.toml << EOF

[tool.flwr.federations.remote-federation]
address = "${SRV_IP}:${SL_CONTROL_PORT}"
insecure = true
EOF
        info "  pyproject.toml → added [tool.flwr.federations.remote-federation]"
    fi

    # Update ~/.flwr/config.toml remote-federation address (Flower 1.29+)
    local flwr_cfg="$HOME/.flwr/config.toml"
    if [ -f "$flwr_cfg" ]; then
        python3 - "$flwr_cfg" "${SRV_IP}:${SL_CONTROL_PORT}" << 'PY'
import sys, re
cfg_path, new_addr = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    content = f.read()
content = re.sub(
    r'(\[superlink\.remote-federation\][^\[]*address\s*=\s*")[^"]*(")',
    lambda m: m.group(1) + new_addr + m.group(2),
    content, flags=re.DOTALL
)
with open(cfg_path, 'w') as f:
    f.write(content)
PY
        info "  ~/.flwr/config.toml → remote-federation.address = ${SRV_IP}:${SL_CONTROL_PORT}"
    fi

    sed -i.bak \
        -e "/^\[tool\.flwr\.federations\.remote-federation\]/,/^\[/ s|^address = \".*\"|address = \"${SRV_IP}:${SL_CONTROL_PORT}\"|" \
        -e "/^\[tool\.flwr\.federations\.remote-federation\]/,/^\[/ s|^insecure = .*|insecure = true|" \
        pyproject.toml
    info "  pyproject.toml → remote-federation.address = ${SRV_IP}:${SL_CONTROL_PORT}"

    # Update [tool.flwr.app.config]
    sed -i.bak \
        -e "s/^num-server-rounds = .*/num-server-rounds = ${NUM_ROUNDS}/" \
        -e "s/^lr = .*/lr = ${LR}/" \
        -e "s/^local-epochs = .*/local-epochs = ${LOCAL_EPOCHS}/" \
        -e "s/^batch-size = .*/batch-size = ${BATCH_SIZE}/" \
        pyproject.toml
    info "  pyproject.toml → rounds=$NUM_ROUNDS  lr=$LR  local-epochs=$LOCAL_EPOCHS  batch-size=$BATCH_SIZE"
    info ""
    info "  Streaming training logs below (Ctrl+C to stop):"
    info "  ServerApp log : outputs/fl_server.log"
    info "  SuperLink log : $LOG_DIR/superlink.log"
    info ""

    export NUM_ROUNDS LOCAL_EPOCHS LR BATCH_SIZE
    export FL_DATA_DIR="$PROJECT_DIR/data/MHEALTHDATASET"

    flwr run . remote-federation \
        --run-config "num-server-rounds=${NUM_ROUNDS} lr=${LR} local-epochs=${LOCAL_EPOCHS} batch-size=${BATCH_SIZE} fraction-train=1.0" \
        --stream \
        2>&1 | tee "$LOG_DIR/training.log"

    local rc="${PIPESTATUS[0]}"
    if [ "$rc" -eq 0 ]; then
        step "TRAINING COMPLETE"
        info "  Results    : outputs/results.json"
        info "  Final model: final_model.pt"
        info "  Dashboard  : http://localhost:$DASHBOARD_PORT/monitor"
        info "  Server log : outputs/fl_server.log"
        activate_venv 2>/dev/null || true
        python3 - << 'PY' 2>/dev/null || true
import json, os
rounds = []
path = "outputs/results.json"
if os.path.exists(path):
    with open(path) as f:
        for line in f:
            try:
                o = json.loads(line.strip())
                if isinstance(o,dict) and o.get("type")=="global":
                    rounds.append(o)
            except Exception: pass
if rounds:
    r = rounds[-1]
    print(f"  Final (round {r['round']}): "
          f"acc={r.get('accuracy',0):.4f}  "
          f"f1={r.get('f1_macro',0):.4f}  "
          f"auc={r.get('auc_macro',0):.4f}  "
          f"loss={r.get('loss',0):.4f}")
else:
    print("  (no global metrics yet — check outputs/fl_server.log)")
PY
    else
        err "flwr run exited with code $rc"
        err "Diagnose: tail $LOG_DIR/training.log"
        err "Server  : tail outputs/fl_server.log"
    fi
    return "$rc"
}

# ─────────────────────────────────────────────────────────────
#  SETUP-PAMAP2 — sync updated code + PAMAP2 data to all Pis
# ─────────────────────────────────────────────────────────────
cmd_setup_pamap2() {
    local PAMAP2_HOSTS=("${PI_HOSTS[@]}")
    step "PAMAP2 SETUP — ${#PAMAP2_HOSTS[@]} Pi(s)"

    wake_pis "${PAMAP2_HOSTS[@]}"

    info "Checking SSH connectivity..."
    for pi in "${PAMAP2_HOSTS[@]}"; do
        check_pi "$pi" \
            && info "  ✓ $pi" \
            || die "Cannot reach $pi. Enable SSH and verify: ssh ${PI_USER}@${pi}"
    done

    # Compute global norm stats on server (needs full dataset, only ~30s)
    local norm_stats="$PROJECT_DIR/data/PAMAP2/Protocol/.norm_stats.npz"
    if [ ! -f "$norm_stats" ]; then
        step "Computing global normalization stats (runs once)"
        activate_venv
        cd "$PROJECT_DIR"
        FL_DATA_DIR="$PROJECT_DIR/data/PAMAP2/Protocol" python3 -c "
from fl_blockchain_evm.core.data import compute_and_save_norm_stats
compute_and_save_norm_stats()
"
        info "  ✓ Norm stats saved → $norm_stats"
    else
        info "  ✓ Norm stats already computed → $norm_stats"
    fi

    step "Syncing code + PAMAP2 data to ${#PAMAP2_HOSTS[@]} Pi(s) in parallel"
    mkdir -p "$LOG_DIR"

    _deploy_pamap2() {
        local pi_host="$1"
        local logf="$LOG_DIR/setup_pamap2_${pi_host}.log"
        {
            echo "[$(ts)] ── PAMAP2 Setup: ${pi_host} ──"

            echo "[$(ts)] Syncing source code..."
            tar -czf - \
                --exclude='venv' --exclude='.git' --exclude='__pycache__' \
                --exclude='*.pyc' --exclude='outputs' --exclude='outputs_pamap2' \
                --exclude='data' --exclude='.env' \
                --exclude='final_model.pt' --exclude='*.bak' \
                -C "$(dirname "$PROJECT_DIR")" "$(basename "$PROJECT_DIR")" \
            | pi_ssh "$pi_host" \
                "cd ~ \
                 && tar -xzf - --warning=no-unknown-keyword 2>/dev/null \
                 && dirname_tar=\$(ls -d ~/FL-Blockchain-EVM* 2>/dev/null | head -1) \
                 && [ -d ~/FL-Blockchain-EVM ] || mv \"\$dirname_tar\" ~/FL-Blockchain-EVM \
                 && chmod +x ~/FL-Blockchain-EVM/fl.sh \
                 && echo '[$(ts)] Code synced'"

            echo "[$(ts)] Syncing PAMAP2 data..."
            local data_src="$PROJECT_DIR/data/PAMAP2/Protocol"
            pi_ssh "$pi_host" "mkdir -p ~/FL-Blockchain-EVM/data/PAMAP2/Protocol/.npy_cache"
            if ls "$data_src"/subject*.dat 1>/dev/null 2>&1; then
                rsync -a --size-only \
                    "$data_src"/subject*.dat \
                    "${PI_USER}@${pi_host}:~/FL-Blockchain-EVM/data/PAMAP2/Protocol/"
                echo "[$(ts)] Synced $(ls "$data_src"/subject*.dat | wc -l | tr -d ' ') subject files."
            else
                echo "[$(ts)] WARNING: no subject*.dat files found in $data_src"
            fi
            if ls "$data_src/.npy_cache"/s*.npy 1>/dev/null 2>&1; then
                rsync -a --size-only \
                    "$data_src/.npy_cache"/s*.npy \
                    "${PI_USER}@${pi_host}:~/FL-Blockchain-EVM/data/PAMAP2/Protocol/.npy_cache/"
                echo "[$(ts)] Synced npy cache."
            fi
            if [ -f "$data_src/.norm_stats.npz" ]; then
                pi_scp -q "$data_src/.norm_stats.npz" \
                    "${PI_USER}@${pi_host}:~/FL-Blockchain-EVM/data/PAMAP2/Protocol/.norm_stats.npz"
                echo "[$(ts)] Synced norm stats."
            fi

            [ -f "$PROJECT_DIR/.env" ] && \
                pi_scp -q "$PROJECT_DIR/.env" "${PI_USER}@${pi_host}:~/FL-Blockchain-EVM/.env" && \
                echo "[$(ts)] .env copied."

            echo "[$(ts)] Disabling WiFi power management (persistent)..."
            pi_ssh "$pi_host" bash << 'WIFIFIX'
sudo iwconfig wlan0 power off 2>/dev/null || true
sudo mkdir -p /etc/NetworkManager/conf.d
printf '[connection]\nwifi.powersave = 2\n' | sudo tee /etc/NetworkManager/conf.d/wifi-pm.conf > /dev/null
printf '#!/bin/sh\niwconfig wlan0 power off 2>/dev/null || true\n' | sudo tee /etc/network/if-up.d/disable-wifi-pm > /dev/null
sudo chmod +x /etc/network/if-up.d/disable-wifi-pm 2>/dev/null || true
WIFIFIX
            echo "[$(ts)] WiFi power management disabled."

            echo "[$(ts)] ✓ ${pi_host} done."
        } 2>&1 | tee "$logf"
    }

    local pids=()
    for pi in "${PAMAP2_HOSTS[@]}"; do
        _deploy_pamap2 "$pi" &
        pids+=($!)
    done

    local ok=true
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { err "${PAMAP2_HOSTS[$i]} failed → $LOG_DIR/setup_pamap2_${PAMAP2_HOSTS[$i]}.log"; ok=false; }
    done
    [ "$ok" = "true" ] || exit 1

    step "PAMAP2 SETUP COMPLETE"
    info "  Run './fl.sh train-pamap2' to start training."
}

# ─────────────────────────────────────────────────────────────
#  TRAIN-PAMAP2 — N_SESSIONS baseline + N_SESSIONS optimized
#                 All sessions write to outputs_pamap2/
# ─────────────────────────────────────────────────────────────
cmd_train_pamap2() {
    local n_sessions="${N_SESSIONS:-5}"
    local PAMAP2_HOSTS=("${PI_HOSTS[@]}")
    local PAMAP2_PARTS="${#PAMAP2_HOSTS[@]}"

    step "FL PAMAP2 — ${n_sessions} baseline + ${n_sessions} optimized sessions  (${PAMAP2_PARTS} clients)"

    local SRV_IP="${SERVER_IP:-$(server_ip)}"
    [ -n "$SRV_IP" ] || die "Could not detect laptop IP.\nSet: SERVER_IP=192.168.x.x ./fl.sh train-pamap2"

    info "  Server IP       : $SRV_IP"
    for i in "${!PAMAP2_HOSTS[@]}"; do
        info "  Pi $i (part. $i)  : ${PAMAP2_HOSTS[$i]}"
    done
    info "  Sessions        : ${n_sessions} baseline + ${n_sessions} optimized"
    info "  Rounds/session  : $NUM_ROUNDS"
    info "  Output dir      : outputs_pamap2/"

    wake_pis "${PAMAP2_HOSTS[@]}"

    for pi in "${PAMAP2_HOSTS[@]}"; do
        check_pi "$pi" || die "$pi unreachable — run './fl.sh setup-pamap2' first"
        info "  ✓ $pi reachable"
    done

    activate_venv
    mkdir -p "$LOG_DIR" "outputs_pamap2"

    export FL_DATA_DIR="$PROJECT_DIR/data/PAMAP2/Protocol"
    export FL_PROJECT_DIR="$PROJECT_DIR"
    export OUTPUT_BASE_DIR="outputs_pamap2"

    _SL_PID="" _DASH_PID=""
    cleanup() {
        step "SHUTDOWN"
        [ -n "$_SL_PID"   ] && kill "$_SL_PID"   2>/dev/null && info "  ✓ SuperLink stopped"  || true
        [ -n "$_DASH_PID" ] && kill "$_DASH_PID"  2>/dev/null && info "  ✓ Dashboard stopped"  || true
        pkill -f "flower-superlink" 2>/dev/null || true
        for pi in "${PAMAP2_HOSTS[@]}"; do
            pi_ssh "$pi" "pkill -f flower-supernode 2>/dev/null; true" 2>/dev/null &
        done
        wait; info "  ✓ Pi processes stopped"
    }
    trap cleanup EXIT

    # ── 1. Dashboard ─────────────────────────────────────────
    step "1/4  Dashboard"
    python run_dashboard.py > "$LOG_DIR/dashboard.log" 2>&1 &
    _DASH_PID=$!
    sleep 2
    kill -0 "$_DASH_PID" 2>/dev/null \
        && info "  ✓ Dashboard → http://localhost:${DASHBOARD_PORT}/monitor" \
        || warn "  Dashboard failed (training continues)"

    # ── 2. SuperLink ─────────────────────────────────────────
    step "2/4  Flower SuperLink"
    pkill -f "flower-superlink" 2>/dev/null || true; sleep 1

    cd "$PROJECT_DIR"
    flower-superlink \
        --insecure \
        --serverappio-api-address "0.0.0.0:${SL_SERVERAPPIO_PORT}" \
        --fleet-api-address       "0.0.0.0:${SL_FLEET_PORT}" \
        --control-api-address     "0.0.0.0:${SL_CONTROL_PORT}" \
        > "$LOG_DIR/superlink.log" 2>&1 &
    _SL_PID=$!
    sleep 3
    kill -0 "$_SL_PID" 2>/dev/null \
        || { err "SuperLink failed to start"; cat "$LOG_DIR/superlink.log"; exit 1; }
    info "  ✓ SuperLink PID=$_SL_PID  Fleet: $SRV_IP:$SL_FLEET_PORT"

    # ── 3. SuperNodes on 7 Pis ───────────────────────────────
    step "3/4  SuperNodes on ${PAMAP2_PARTS} Pi(s)"

    _start_pamap2_supernode() {
        local pi_host="$1" part_id="$2"
        local logf="$LOG_DIR/supernode_pamap2_${part_id}.log"

        local script
        script=$(cat << PIRUN
set -e
LOGF="/tmp/fl_client_pamap2_${part_id}.log"
: > "\$LOGF"
_l() { echo "[\$(date '+%Y-%m-%d %H:%M:%S')] \$*" | tee -a "\$LOGF"; }

_l "════════════════════════════════════════════════════"
_l "  FL SuperNode (PAMAP2) — Partition ${part_id} / ${PAMAP2_PARTS}"
_l "  Host      : \$(hostname)  (${pi_host})"
_l "  Server    : ${SRV_IP}:${SL_FLEET_PORT}"
_l "  DATA_DIR  : \$HOME/FL-Blockchain-EVM/data/PAMAP2/Protocol"
_l "════════════════════════════════════════════════════"

cd ~/FL-Blockchain-EVM
source venv/bin/activate
pkill -f flower-supernode 2>/dev/null && _l "Killed stale supernode" || true
sleep 1

export FL_DATA_DIR="\$HOME/FL-Blockchain-EVM/data/PAMAP2/Protocol"

_l "Starting flower-supernode (detached from SSH)..."
nohup flower-supernode \\
    --insecure \\
    --superlink "${SRV_IP}:${SL_FLEET_PORT}" \\
    --node-config "partition-id=${part_id} num-partitions=${PAMAP2_PARTS}" \\
    --clientappio-api-address "0.0.0.0:${SN_CLIENTAPPIO_PORT}" \\
    >> "\$LOGF" 2>&1 &
disown \$!
_l "SuperNode running (PID=\$!), detached from SSH."
PIRUN
)
        echo "$script" | pi_ssh "$pi_host" bash > "$logf" 2>&1 &
        sleep 2
        info "  ✓ SuperNode ${part_id} started on ${pi_host}"
        info "    Remote log : ssh ${PI_USER}@${pi_host} tail -f /tmp/fl_client_pamap2_${part_id}.log"
    }

    for i in "${!PAMAP2_HOSTS[@]}"; do
        _start_pamap2_supernode "${PAMAP2_HOSTS[$i]}" "$i"
    done

    info ""
    info "  Waiting 15s for SuperNodes to register..."
    sleep 15

    # Update pyproject.toml address
    if ! grep -q '^\[tool\.flwr\.federations\]' pyproject.toml; then
        printf '\n[tool.flwr.federations]\ndefault = "remote-federation"\n' >> pyproject.toml
    fi
    if ! grep -q '^\[tool\.flwr\.federations\.remote-federation\]' pyproject.toml; then
        printf '\n[tool.flwr.federations.remote-federation]\naddress = "%s:%s"\ninsecure = true\n' \
            "$SRV_IP" "$SL_CONTROL_PORT" >> pyproject.toml
    fi
    sed -i.bak \
        -e "/^\[tool\.flwr\.federations\.remote-federation\]/,/^\[/ s|^address = \".*\"|address = \"${SRV_IP}:${SL_CONTROL_PORT}\"|" \
        -e "/^\[tool\.flwr\.federations\.remote-federation\]/,/^\[/ s|^insecure = .*|insecure = true|" \
        pyproject.toml
    sed -i.bak \
        -e "s/^num-server-rounds = .*/num-server-rounds = ${NUM_ROUNDS}/" \
        -e "s/^lr = .*/lr = ${LR}/" \
        -e "s/^local-epochs = .*/local-epochs = ${LOCAL_EPOCHS}/" \
        -e "s/^batch-size = .*/batch-size = ${BATCH_SIZE}/" \
        -e "s/^num-partitions = .*/num-partitions = ${PAMAP2_PARTS}/" \
        pyproject.toml
    info "  pyproject.toml updated (rounds=$NUM_ROUNDS  partitions=$PAMAP2_PARTS)"

    # ── 4. Session loop ──────────────────────────────────────
    step "4/4  Session loop"

    _run_pamap2_session() {
        local variant="$1" sess="$2" bc_opt="$3"
        step "SESSION ${sess}/${n_sessions}  variant=${variant}  BLOCKCHAIN_OPTIMIZED=${bc_opt}"
        export EXPERIMENT_VARIANT="$variant"
        export BLOCKCHAIN_OPTIMIZED="$bc_opt"

        flwr run . remote-federation \
            --run-config "num-server-rounds=${NUM_ROUNDS} lr=${LR} local-epochs=${LOCAL_EPOCHS} batch-size=${BATCH_SIZE} fraction-train=1.0 num-partitions=${PAMAP2_PARTS} experiment-variant=\"${variant}\" blockchain-optimized=${bc_opt}" \
            --stream \
            2>&1 | tee "$LOG_DIR/training_${variant}_s${sess}.log"

        local rc="${PIPESTATUS[0]}"
        if [ "$rc" -eq 0 ]; then
            info "  ✓ Session ${sess} (${variant}) complete."
        else
            err "  Session ${sess} (${variant}) exited with code $rc"
        fi

        info "  Sleeping 15s between sessions..."
        sleep 15
    }

    step "ALTERNATING PHASE  (${n_sessions} baseline + ${n_sessions} optimized, interleaved)"
    for s in $(seq 1 "$n_sessions"); do
        _run_pamap2_session "baseline"  "$s" "0"
        _run_pamap2_session "optimized" "$s" "1"
    done

    step "ALL PAMAP2 SESSIONS COMPLETE"
    info "  Output: outputs_pamap2/"
    activate_venv 2>/dev/null || true
    python3 - "outputs_pamap2" << 'PY' 2>/dev/null || true
import json, os, sys, glob
base = sys.argv[1]
sessions = sorted(glob.glob(f"{base}/*/results.json"))
if not sessions:
    print("  (no completed sessions found)")
else:
    for path in sessions:
        folder = os.path.basename(os.path.dirname(path))
        rounds = []
        with open(path) as f:
            for line in f:
                try:
                    o = json.loads(line.strip())
                    if isinstance(o, dict) and o.get("type") == "global":
                        rounds.append(o)
                except Exception:
                    pass
        if rounds:
            r = rounds[-1]
            print(f"  {folder:<40}  acc={r.get('accuracy',0):.4f}  "
                  f"f1={r.get('f1_macro',0):.4f}  auc={r.get('auc_macro',0):.4f}")
PY
}

# ─────────────────────────────────────────────────────────────
#  SETUP-UCIHAR — sync code + UCI HAR data to all Pis
# ─────────────────────────────────────────────────────────────
cmd_setup_ucihar() {
    local UCI_HOSTS=("${PI_HOSTS[@]}")
    step "UCI HAR SETUP — ${#UCI_HOSTS[@]} Pi(s)"

    wake_pis "${UCI_HOSTS[@]}"

    info "Checking SSH connectivity..."
    for pi in "${UCI_HOSTS[@]}"; do
        check_pi "$pi" \
            && info "  ✓ $pi" \
            || die "Cannot reach $pi. Verify: ssh ${PI_USER}@${pi}"
    done

    local uci_data="$PROJECT_DIR/data/UCI_HAR/UCI_HAR_Dataset"
    local norm_stats="$uci_data/.norm_stats.npz"
    if [ ! -f "$norm_stats" ]; then
        step "Computing global normalization stats (runs once)"
        activate_venv
        cd "$PROJECT_DIR"
        FL_DATASET=ucihar FL_DATA_DIR="$uci_data" python3 -c "
from fl_blockchain_evm.core.data_ucihar import compute_and_save_norm_stats
compute_and_save_norm_stats()
"
        info "  ✓ Norm stats saved → $norm_stats"
    else
        info "  ✓ Norm stats already computed → $norm_stats"
    fi

    step "Syncing code + UCI HAR data to ${#UCI_HOSTS[@]} Pi(s) in parallel"
    mkdir -p "$LOG_DIR"

    _deploy_ucihar() {
        local pi_host="$1"
        local logf="$LOG_DIR/setup_ucihar_${pi_host}.log"
        {
            echo "[$(ts)] ── UCI HAR Setup: ${pi_host} ──"

            echo "[$(ts)] Syncing source code..."
            tar -czf - \
                --exclude='venv' --exclude='.git' --exclude='__pycache__' \
                --exclude='*.pyc' --exclude='outputs' --exclude='outputs_pamap2' \
                --exclude='outputs_ucihar' --exclude='data' --exclude='.env' \
                --exclude='final_model.pt' --exclude='*.bak' \
                -C "$(dirname "$PROJECT_DIR")" "$(basename "$PROJECT_DIR")" \
            | pi_ssh "$pi_host" \
                "cd ~ \
                 && tar -xzf - --warning=no-unknown-keyword 2>/dev/null \
                 && [ -d ~/FL-Blockchain-EVM ] || mv ~/FL-Blockchain-EVM* ~/FL-Blockchain-EVM \
                 && chmod +x ~/FL-Blockchain-EVM/fl.sh \
                 && echo 'Code synced'"

            echo "[$(ts)] Syncing UCI HAR data..."
            pi_ssh "$pi_host" "mkdir -p ~/FL-Blockchain-EVM/data/UCI_HAR/UCI_HAR_Dataset/train/Inertial\ Signals ~/FL-Blockchain-EVM/data/UCI_HAR/UCI_HAR_Dataset/test/Inertial\ Signals"
            rsync -a --size-only \
                "$uci_data/train/" \
                "${PI_USER}@${pi_host}:FL-Blockchain-EVM/data/UCI_HAR/UCI_HAR_Dataset/train/"
            rsync -a --size-only \
                "$uci_data/test/" \
                "${PI_USER}@${pi_host}:FL-Blockchain-EVM/data/UCI_HAR/UCI_HAR_Dataset/test/"
            if [ -f "$uci_data/.norm_stats.npz" ]; then
                pi_scp -q "$uci_data/.norm_stats.npz" \
                    "${PI_USER}@${pi_host}:FL-Blockchain-EVM/data/UCI_HAR/UCI_HAR_Dataset/.norm_stats.npz"
            fi
            echo "[$(ts)] UCI HAR data synced."

            [ -f "$PROJECT_DIR/.env" ] && \
                pi_scp -q "$PROJECT_DIR/.env" "${PI_USER}@${pi_host}:~/FL-Blockchain-EVM/.env"

            pi_ssh "$pi_host" bash << 'WIFIFIX'
sudo iwconfig wlan0 power off 2>/dev/null || true
sudo mkdir -p /etc/NetworkManager/conf.d
printf '[connection]\nwifi.powersave = 2\n' | sudo tee /etc/NetworkManager/conf.d/wifi-pm.conf > /dev/null
printf '#!/bin/sh\niwconfig wlan0 power off 2>/dev/null || true\n' | sudo tee /etc/network/if-up.d/disable-wifi-pm > /dev/null
sudo chmod +x /etc/network/if-up.d/disable-wifi-pm 2>/dev/null || true
WIFIFIX
            echo "[$(ts)] ✓ ${pi_host} done."
        } 2>&1 | tee "$logf"
    }

    local pids=()
    for pi in "${UCI_HOSTS[@]}"; do
        _deploy_ucihar "$pi" &
        pids+=($!)
    done
    local ok=true
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { err "${UCI_HOSTS[$i]} failed → $LOG_DIR/setup_ucihar_${UCI_HOSTS[$i]}.log"; ok=false; }
    done
    [ "$ok" = "true" ] || exit 1

    step "UCI HAR SETUP COMPLETE"
    info "  Run './fl.sh train-ucihar' to start training."
}

# ─────────────────────────────────────────────────────────────
#  TRAIN-UCIHAR — N_SESSIONS baseline + N_SESSIONS optimized
#                 All sessions write to outputs_ucihar/
# ─────────────────────────────────────────────────────────────
cmd_train_ucihar() {
    local n_sessions="${N_SESSIONS:-10}"
    local UCI_HOSTS=("${PI_HOSTS[@]}")
    local UCI_PARTS="${#UCI_HOSTS[@]}"
    local UCI_ROUNDS="${NUM_ROUNDS:-20}"
    local UCI_DATA="$PROJECT_DIR/data/UCI_HAR/UCI_HAR_Dataset"

    step "FL UCI HAR — ${n_sessions}+${n_sessions} sessions  (${UCI_PARTS} clients, ${UCI_ROUNDS} rounds)"

    local SRV_IP="${SERVER_IP:-$(server_ip)}"
    [ -n "$SRV_IP" ] || die "Could not detect server IP. Set SERVER_IP=..."

    info "  Server IP  : $SRV_IP"
    for i in "${!UCI_HOSTS[@]}"; do info "  Pi $i  : ${UCI_HOSTS[$i]}"; done
    info "  Sessions   : ${n_sessions} baseline + ${n_sessions} optimized"
    info "  Rounds     : $UCI_ROUNDS  |  LR: $LR  |  Epochs: $LOCAL_EPOCHS"
    info "  Output dir : outputs_ucihar/"

    wake_pis "${UCI_HOSTS[@]}"
    for pi in "${UCI_HOSTS[@]}"; do
        check_pi "$pi" || die "$pi unreachable — run './fl.sh setup-ucihar' first"
        info "  ✓ $pi reachable"
    done

    activate_venv
    mkdir -p "$LOG_DIR" "outputs_ucihar"

    export FL_DATASET="ucihar"
    export FL_DATA_DIR="$UCI_DATA"
    export OUTPUT_BASE_DIR="outputs_ucihar"

    _SL_PID="" _DASH_PID=""
    cleanup() {
        step "SHUTDOWN"
        [ -n "$_SL_PID"  ] && kill "$_SL_PID"  2>/dev/null && info "  ✓ SuperLink stopped" || true
        [ -n "$_DASH_PID"] && kill "$_DASH_PID" 2>/dev/null && info "  ✓ Dashboard stopped" || true
        pkill -f "flower-superlink" 2>/dev/null || true
        for pi in "${UCI_HOSTS[@]}"; do
            pi_ssh "$pi" "pkill -f flower-supernode 2>/dev/null; true" 2>/dev/null &
        done
        wait; info "  ✓ Pi processes stopped"
    }
    trap cleanup EXIT

    step "1/4  Dashboard"
    python run_dashboard.py > "$LOG_DIR/dashboard.log" 2>&1 &
    _DASH_PID=$!
    sleep 2
    kill -0 "$_DASH_PID" 2>/dev/null \
        && info "  ✓ Dashboard → http://localhost:${DASHBOARD_PORT}/monitor" \
        || warn "  Dashboard failed (training continues)"

    step "2/4  Flower SuperLink"
    pkill -f "flower-superlink" 2>/dev/null || true; sleep 1
    cd "$PROJECT_DIR"
    flower-superlink \
        --insecure \
        --serverappio-api-address "0.0.0.0:${SL_SERVERAPPIO_PORT}" \
        --fleet-api-address       "0.0.0.0:${SL_FLEET_PORT}" \
        --control-api-address     "0.0.0.0:${SL_CONTROL_PORT}" \
        > "$LOG_DIR/superlink.log" 2>&1 &
    _SL_PID=$!
    sleep 3
    kill -0 "$_SL_PID" 2>/dev/null \
        || { err "SuperLink failed"; cat "$LOG_DIR/superlink.log"; exit 1; }
    info "  ✓ SuperLink PID=$_SL_PID  Fleet: $SRV_IP:$SL_FLEET_PORT"

    step "3/4  SuperNodes on ${UCI_PARTS} Pi(s)"

    _start_ucihar_supernode() {
        local pi_host="$1" part_id="$2"
        local script
        script=$(cat << PIRUN
set -e
LOGF="/tmp/fl_client_ucihar_${part_id}.log"
: > "\$LOGF"
_l() { echo "[\$(date '+%Y-%m-%d %H:%M:%S')] \$*" | tee -a "\$LOGF"; }
_l "UCI HAR SuperNode — Partition ${part_id}/${UCI_PARTS}  host=\$(hostname)  RAM=\$(free -m | awk '/^Mem/{print \$2}')MB"
cd ~/FL-Blockchain-EVM
source venv/bin/activate
pkill -f flower-supernode 2>/dev/null && _l "Killed stale supernode" || true
sleep 1
export FL_DATASET="ucihar"
export FL_DATA_DIR="\$HOME/FL-Blockchain-EVM/data/UCI_HAR/UCI_HAR_Dataset"
nohup flower-supernode \\
    --insecure \\
    --superlink "${SRV_IP}:${SL_FLEET_PORT}" \\
    --node-config "partition-id=${part_id} num-partitions=${UCI_PARTS}" \\
    --clientappio-api-address "0.0.0.0:${SN_CLIENTAPPIO_PORT}" \\
    >> "\$LOGF" 2>&1 &
disown \$!
_l "SuperNode running (PID=\$!)"
PIRUN
)
        echo "$script" | pi_ssh "$pi_host" bash > "$LOG_DIR/supernode_ucihar_${part_id}.log" 2>&1 &
        sleep 2
        info "  ✓ SuperNode ${part_id} on ${pi_host}  (log: ssh ${PI_USER}@${pi_host} tail -f /tmp/fl_client_ucihar_${part_id}.log)"
    }

    for i in "${!UCI_HOSTS[@]}"; do
        _start_ucihar_supernode "${UCI_HOSTS[$i]}" "$i"
    done
    info "  Waiting 15s for SuperNodes to register..."
    sleep 15

    if ! grep -q '^\[tool\.flwr\.federations\]' pyproject.toml; then
        printf '\n[tool.flwr.federations]\ndefault = "remote-federation"\n' >> pyproject.toml
    fi
    if ! grep -q '^\[tool\.flwr\.federations\.remote-federation\]' pyproject.toml; then
        printf '\n[tool.flwr.federations.remote-federation]\naddress = "%s:%s"\ninsecure = true\n' \
            "$SRV_IP" "$SL_CONTROL_PORT" >> pyproject.toml
    fi
    sed -i.bak \
        -e "/^\[tool\.flwr\.federations\.remote-federation\]/,/^\[/ s|^address = \".*\"|address = \"${SRV_IP}:${SL_CONTROL_PORT}\"|" \
        -e "/^\[tool\.flwr\.federations\.remote-federation\]/,/^\[/ s|^insecure = .*|insecure = true|" \
        pyproject.toml
    sed -i.bak \
        -e "s/^num-server-rounds = .*/num-server-rounds = ${UCI_ROUNDS}/" \
        -e "s/^lr = .*/lr = ${LR}/" \
        -e "s/^local-epochs = .*/local-epochs = ${LOCAL_EPOCHS}/" \
        -e "s/^batch-size = .*/batch-size = ${BATCH_SIZE}/" \
        -e "s/^num-partitions = .*/num-partitions = ${UCI_PARTS}/" \
        pyproject.toml
    info "  pyproject.toml updated (rounds=$UCI_ROUNDS  partitions=$UCI_PARTS)"

    step "4/4  Session loop"

    _run_ucihar_session() {
        local variant="$1" sess="$2" bc_opt="$3"
        step "SESSION ${sess}/${n_sessions}  variant=${variant}"
        export EXPERIMENT_VARIANT="$variant"
        export BLOCKCHAIN_OPTIMIZED="$bc_opt"
        export FL_DATASET="ucihar"
        export FL_DATA_DIR="$UCI_DATA"
        export OUTPUT_BASE_DIR="outputs_ucihar"

        flwr run . remote-federation \
            --run-config "num-server-rounds=${UCI_ROUNDS} lr=${LR} local-epochs=${LOCAL_EPOCHS} batch-size=${BATCH_SIZE} fraction-train=1.0 num-partitions=${UCI_PARTS} experiment-variant=\"${variant}\" blockchain-optimized=${bc_opt}" \
            --stream \
            2>&1 | tee "$LOG_DIR/training_ucihar_${variant}_s${sess}.log"

        local rc="${PIPESTATUS[0]}"
        [ "$rc" -eq 0 ] && info "  ✓ Session ${sess} (${variant}) complete." \
                        || err "  Session ${sess} (${variant}) exited with code $rc"
        info "  Sleeping 15s..."; sleep 15
    }

    step "ALTERNATING PHASE  (${n_sessions} baseline + ${n_sessions} optimized, interleaved)"
    for s in $(seq 1 "$n_sessions"); do
        _run_ucihar_session "baseline"  "$s" "0"
        _run_ucihar_session "optimized" "$s" "1"
    done

    step "ALL UCI HAR SESSIONS COMPLETE"
    info "  Output: outputs_ucihar/"
    activate_venv 2>/dev/null || true
    python3 - "outputs_ucihar" << 'PY' 2>/dev/null || true
import json, os, sys, glob
base = sys.argv[1]
sessions = sorted(glob.glob(f"{base}/*/results.json"))
if not sessions:
    print("  (no completed sessions found)")
else:
    for path in sessions:
        folder = os.path.basename(os.path.dirname(path))
        rounds = []
        with open(path) as f:
            for line in f:
                try:
                    o = json.loads(line.strip())
                    if isinstance(o, dict) and o.get("type") == "global":
                        rounds.append(o)
                except Exception:
                    pass
        if rounds:
            r = rounds[-1]
            print(f"  {folder:<40}  acc={r.get('accuracy',0):.4f}  "
                  f"f1={r.get('f1_macro',0):.4f}  auc={r.get('auc_macro',0):.4f}")
PY
}

# ─────────────────────────────────────────────────────────────
#  STOP
# ─────────────────────────────────────────────────────────────
cmd_stop() {
    step "Stopping all FL processes"
    pkill -f "flower-superlink"  2>/dev/null && info "  ✓ SuperLink"   || info "  - SuperLink not running"
    pkill -f "flwr run"          2>/dev/null && info "  ✓ flwr run"    || true
    pkill -f "run_dashboard"     2>/dev/null && info "  ✓ Dashboard"   || info "  - Dashboard not running"
    pkill -f "uvicorn"           2>/dev/null && info "  ✓ uvicorn"     || true
    for pi in "${PI_HOSTS[@]}"; do
        if check_pi "$pi" 2>/dev/null; then
            pi_ssh "$pi" "pkill -f flower-supernode 2>/dev/null; true" 2>/dev/null \
                && info "  ✓ $pi SuperNode stopped" || true
        else
            warn "  $pi unreachable — skipped"
        fi
    done
    info ""; info "✓ Done."
}

# ─────────────────────────────────────────────────────────────
#  LOGS — live log aggregation from all devices
# ─────────────────────────────────────────────────────────────
cmd_logs() {
    step "Live Logs — all devices  (Ctrl+C to exit)"
    mkdir -p "$LOG_DIR"
    for f in training superlink dashboard; do touch "$LOG_DIR/${f}.log"; done
    touch "outputs/fl_server.log" 2>/dev/null || true

    for i in "${!PI_HOSTS[@]}"; do
        local pi_host="${PI_HOSTS[$i]}"
        if check_pi "$pi_host" 2>/dev/null; then
            (pi_ssh "$pi_host" \
                "tail -n 40 -f /tmp/fl_client_${i}.log 2>/dev/null || echo 'no log yet'" \
                2>/dev/null \
                | sed "s/^/${C}[Pi${i} ${pi_host}]${N} /") &
        else
            warn "  $pi_host unreachable"
        fi
    done

    tail -n 20 -f \
        "$LOG_DIR/training.log" \
        "$LOG_DIR/superlink.log" \
        "outputs/fl_server.log" \
    2>/dev/null \
    | awk -v g="$G" -v y="$Y" -v c="$C" -v n="$N" '
        /==> .* training\.log/   { src=g"[TRAINING]  "n; next }
        /==> .* superlink\.log/  { src=y"[SUPERLINK] "n; next }
        /==> .* fl_server\.log/  { src=c"[SERVER]    "n; next }
        { print src $0 }
    ' &

    wait
}

# ─────────────────────────────────────────────────────────────
#  STATUS
# ─────────────────────────────────────────────────────────────
cmd_status() {
    step "FL System Status  [$(ts)]"
    echo ""
    echo "─── Laptop ──────────────────────────────────────────"
    pgrep -f "flower-superlink" >/dev/null 2>&1 \
        && echo "  ✓ SuperLink    running" || echo "  ✗ SuperLink    stopped"
    pgrep -f "flwr run"         >/dev/null 2>&1 \
        && echo "  ✓ flwr run     running" || echo "  ✗ flwr run     stopped"
    pgrep -f "run_dashboard"    >/dev/null 2>&1 \
        && echo "  ✓ Dashboard    running → http://localhost:${DASHBOARD_PORT}/monitor" \
        || echo "  ✗ Dashboard    stopped"
    echo ""
    echo "─── Pis (${NUM_PARTITIONS} total) ───────────────────────────────────"
    for i in "${!PI_HOSTS[@]}"; do
        local pi_host="${PI_HOSTS[$i]}"
        if check_pi "$pi_host" 2>/dev/null; then
            local st
            st=$(pi_ssh "$pi_host" \
                "pgrep -fa flower-supernode 2>/dev/null | head -1 || echo 'stopped'" 2>/dev/null)
            echo "$st" | grep -q "flower-supernode" \
                && echo "  ✓ Pi${i} ($pi_host)  SuperNode running" \
                || echo "  ✗ Pi${i} ($pi_host)  SuperNode stopped"
        else
            echo "  ✗ Pi${i} ($pi_host)  UNREACHABLE"
        fi
    done
    echo ""
    echo "─── Training ────────────────────────────────────────"
    activate_venv 2>/dev/null || true
    python3 - << 'PY' 2>/dev/null || echo "  (no results yet)"
import json, os
path = "outputs/results.json"
if not os.path.exists(path):
    print("  No results — run './fl.sh train'")
else:
    rounds = []
    with open(path) as f:
        for line in f:
            try:
                o = json.loads(line.strip())
                if isinstance(o,dict) and o.get("type")=="global":
                    rounds.append(o)
            except Exception: pass
    if not rounds:
        print("  Training in progress (no completed rounds yet)...")
    else:
        print(f"  Completed rounds : {len(rounds)}")
        r = rounds[-1]
        print(f"  Latest (round {r['round']:2d}) : "
              f"acc={r.get('accuracy',0):.4f}  "
              f"f1={r.get('f1_macro',0):.4f}  "
              f"auc={r.get('auc_macro',0):.4f}  "
              f"loss={r.get('loss',0):.4f}")
        if len(rounds) > 1:
            da = rounds[-1].get('accuracy',0) - rounds[0].get('accuracy',0)
            df = rounds[-1].get('f1_macro',0)  - rounds[0].get('f1_macro',0)
            print(f"  Trend            : acc{'+' if da>=0 else ''}{da:.4f}  "
                  f"f1{'+' if df>=0 else ''}{df:.4f}")
PY
    echo ""
    echo "─── Logs ────────────────────────────────────────────"
    echo "  ./fl.sh logs"
    for i in "${!PI_HOSTS[@]}"; do
        echo "  ssh ${PI_USER}@${PI_HOSTS[$i]} tail -f /tmp/fl_client_${i}.log"
    done
    echo "  tail -f $LOG_DIR/superlink.log"
    echo "  tail -f outputs/fl_server.log"
    echo ""
}

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
CMD="${1:-help}"; shift 2>/dev/null || true
case "$CMD" in
    setup)          cmd_setup          ;;
    train)          cmd_train          ;;
    setup-pamap2)   cmd_setup_pamap2   ;;
    train-pamap2)   cmd_train_pamap2   ;;
    setup-ucihar)   cmd_setup_ucihar   ;;
    train-ucihar)   cmd_train_ucihar   ;;
    stop)           cmd_stop           ;;
    logs)           cmd_logs           ;;
    status)         cmd_status         ;;
    *)
        echo -e ""
        echo -e "${B}FL — Federated Learning (${NUM_PARTITIONS} × Raspberry Pi 4)${N}"
        echo -e ""
        echo -e "Usage: ${B}./fl.sh <command>${N}"
        echo -e ""
        echo -e "${C}── MHEALTH ──────────────────────────────────────────────${N}"
        printf "  ${G}%-16s${N}  %s\n" "setup"         "Deploy MHEALTH code + data to all Pis, install deps"
        printf "  ${G}%-16s${N}  %s\n" "train"         "Run one FL session (MHEALTH, outputs/)"
        echo -e ""
        echo -e "${C}── UCI HAR ──────────────────────────────────────────────${N}"
        printf "  ${G}%-16s${N}  %s\n" "setup-ucihar"  "Sync UCI HAR code + data to all Pis"
        printf "  ${G}%-16s${N}  %s\n" "train-ucihar"  "Run N_SESSIONS baseline + N_SESSIONS optimized (outputs_ucihar/)"
        echo -e ""
        echo -e "${C}── PAMAP2 ───────────────────────────────────────────────${N}"
        printf "  ${G}%-16s${N}  %s\n" "setup-pamap2"  "Sync PAMAP2 code + data to all Pis"
        printf "  ${G}%-16s${N}  %s\n" "train-pamap2"  "Run N_SESSIONS baseline + N_SESSIONS optimized (outputs_pamap2/)"
        echo -e ""
        echo -e "${C}── GENERAL ──────────────────────────────────────────────${N}"
        printf "  ${G}%-16s${N}  %s\n" "stop"          "Stop all FL processes on laptop and all Pis"
        printf "  ${G}%-16s${N}  %s\n" "logs"          "Stream live logs from all devices simultaneously"
        printf "  ${G}%-16s${N}  %s\n" "status"        "Show training progress and system health"
        echo -e ""
        echo -e "Env overrides:  PI_USER=pi  NUM_ROUNDS=10  N_SESSIONS=5  LOCAL_EPOCHS=1  BATCH_SIZE=256  LR=0.002  SERVER_IP=..."
        echo -e ""
        echo -e "Pis configured (${NUM_PARTITIONS} total):"
        for i in "${!PI_HOSTS[@]}"; do
            echo -e "  Pi${i} → ${PI_HOSTS[$i]}  (partition $i)"
        done
        echo -e ""
        echo -e "PAMAP2 workflow: ${C}./fl.sh setup-pamap2${N} → ${C}./fl.sh train-pamap2${N}"
        echo -e "MHEALTH workflow: ${C}./fl.sh setup${N} → ${C}./fl.sh train${N}"
        echo ""
        ;;
esac
