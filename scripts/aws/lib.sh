#!/usr/bin/env bash
# Shared helpers for scripts/aws/*.sh  (source this file; do not execute it).

set -euo pipefail

AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$AWS_DIR/../.." && pwd)"
STATE_FILE="$AWS_DIR/.state"
ENV_FILE="$AWS_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
elif [[ -f "$AWS_DIR/config.env.example" ]]; then
  # shellcheck disable=SC1091
  source "$AWS_DIR/config.env.example"
fi

AWS_REGION="${AWS_REGION:-us-east-1}"
MODE="${MODE:-cheap}"
VOLUME_GB="${VOLUME_GB:-100}"
NAME="${NAME:-hapi-gpu}"
AMI_ID="${AMI_ID:-}"
SSH_CIDR="${SSH_CIDR:-}"
REQUEST_GPU_QUOTA="${REQUEST_GPU_QUOTA:-1}"
BUDGET_LIMIT="${BUDGET_LIMIT:-25}"
SSH_USER="${SSH_USER:-ubuntu}"
KEY_NAME="${KEY_NAME:-hapi-aws-${AWS_REGION}}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/${KEY_NAME}.pem}"
SG_NAME="${SG_NAME:-hapi-ssh-${AWS_REGION}}"
SUITE="${SUITE:-full}"
EVALUATION_BOOTSTRAP="${EVALUATION_BOOTSTRAP:-2000}"
AGGREGATE_BOOTSTRAP="${AGGREGATE_BOOTSTRAP:-10000}"
SINGLE8_INSTANCE_TYPE="${SINGLE8_INSTANCE_TYPE:-g5.48xlarge}"

ALL_VARIANTS=(
  A0_unet2d_t2
  A1_resunet2d_t2
  A2_resunet2p5d_t2
  A3_resunet2d_nested
  A4_proposed
  A5_independent_heads
  A6_transformer
  A7_ordinal_rps
  A8_no_brier
)
# 8 GPUs / 9 variants: A6 (slowest) stays alone; A7+A8 share GPU 7.
SINGLE8_VARIANT_GROUPS=(
  A0_unet2d_t2
  A1_resunet2d_t2
  A2_resunet2p5d_t2
  A3_resunet2d_nested
  A4_proposed
  A5_independent_heads
  A6_transformer
  A7_ordinal_rps,A8_no_brier
)

export AWS_DEFAULT_REGION="$AWS_REGION"

apply_runtime_mode() {
  case "$MODE" in
    cheap)
      INSTANCE_TYPE="g4dn.xlarge"
      INSTANCE_COUNT=1
      GPUS_PER_INSTANCE=1
      MIN_VCPU_QUOTA=4
      QUOTA_REQUEST=8
      DATALOADER_WORKERS=2
      MODE_TITLE="cheap: 1x g4dn.xlarge (T4), sequential"
      MODE_HINT="Lowest \$/hour. Full suite is many days on one GPU."
      ;;
    parallel9)
      INSTANCE_TYPE="g5.xlarge"
      INSTANCE_COUNT=9
      GPUS_PER_INSTANCE=1
      MIN_VCPU_QUOTA=36
      QUOTA_REQUEST=40
      DATALOADER_WORKERS=2
      MODE_TITLE="parallel9: 9x g5.xlarge (A10G), one variant each"
      MODE_HINT="Fastest wall-clock among these modes. Needs ~36 G/VT vCPUs and 9 public IPs."
      ;;
    single8)
      INSTANCE_TYPE="$SINGLE8_INSTANCE_TYPE"
      INSTANCE_COUNT=1
      GPUS_PER_INSTANCE=8
      MIN_VCPU_QUOTA=192
      QUOTA_REQUEST=192
      DATALOADER_WORKERS=2
      MODE_TITLE="single8: 1x ${SINGLE8_INSTANCE_TYPE} (8 GPUs on one box)"
      MODE_HINT="One machine, 8 workers. Default g5.48xlarge needs ~192 G/VT vCPUs. Set SINGLE8_INSTANCE_TYPE=p4d.24xlarge for A100s (different quota)."
      ;;
    *)
      echo "Unknown MODE='$MODE'. Use cheap, parallel9, or single8." >&2
      exit 1
      ;;
  esac
}

take_mode_arg() {
  if [[ "${1:-}" == "cheap" || "${1:-}" == "parallel9" || "${1:-}" == "single8" ]]; then
    MODE="$1"
    apply_runtime_mode
    return 0
  fi
  return 1
}

apply_runtime_mode

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1" >&2
    exit 1
  }
}

print_mode_banner() {
  echo "==> MODE=$MODE"
  echo "    $MODE_TITLE"
  echo "    $MODE_HINT"
  echo "    instance_type=$INSTANCE_TYPE count=$INSTANCE_COUNT gpus/instance=$GPUS_PER_INSTANCE"
  echo "    suite=$SUITE  (set SUITE=screen for the 45-run seed-42 matrix)"
}

load_state() {
  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  fi
}

save_state() {
  umask 077
  cat >"$STATE_FILE" <<EOF
MODE=$(printf '%q' "$MODE")
INSTANCE_TYPE=$(printf '%q' "$INSTANCE_TYPE")
INSTANCE_COUNT=$INSTANCE_COUNT
INSTANCE_IDS=$(printf '%q' "${INSTANCE_IDS:-}")
PUBLIC_IPS=$(printf '%q' "${PUBLIC_IPS:-}")
INSTANCE_ID=$(printf '%q' "${INSTANCE_ID:-}")
PUBLIC_IP=$(printf '%q' "${PUBLIC_IP:-}")
SG_ID=$(printf '%q' "${SG_ID:-}")
KEY_PATH=$(printf '%q' "$KEY_PATH")
KEY_NAME=$(printf '%q' "$KEY_NAME")
AWS_REGION=$(printf '%q' "$AWS_REGION")
SSH_USER=$(printf '%q' "$SSH_USER")
EOF
}

instance_id_list() {
  load_state
  local ids="${INSTANCE_IDS:-${INSTANCE_ID:-}}"
  # shellcheck disable=SC2206
  INSTANCE_ID_ARR=($ids)
  if [[ ${#INSTANCE_ID_ARR[@]} -eq 0 || -z "${INSTANCE_ID_ARR[0]:-}" ]]; then
    echo "No instances in $STATE_FILE. Run scripts/aws/02_launch_gpu.sh first." >&2
    exit 1
  fi
}

require_instance() {
  instance_id_list
  INSTANCE_ID="${INSTANCE_ID_ARR[0]}"
}

refresh_public_ips() {
  instance_id_list
  PUBLIC_IPS=""
  local id ip
  for id in "${INSTANCE_ID_ARR[@]}"; do
    ip="$(aws ec2 describe-instances \
      --instance-ids "$id" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' \
      --output text)"
    if [[ -z "$ip" || "$ip" == "None" ]]; then
      echo "Instance $id has no public IP (is it stopped?)." >&2
      exit 1
    fi
    PUBLIC_IPS+="$ip "
  done
  PUBLIC_IPS="${PUBLIC_IPS%% }"
  INSTANCE_IDS="${INSTANCE_ID_ARR[*]}"
  INSTANCE_ID="${INSTANCE_ID_ARR[0]}"
  PUBLIC_IP="${PUBLIC_IPS%% *}"
  save_state
}

ssh_opts() {
  echo -i "$KEY_PATH" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o ServerAliveInterval=30
}

ip_for_index() {
  local idx="${1:-0}"
  refresh_public_ips
  # shellcheck disable=SC2206
  local ips=($PUBLIC_IPS)
  if [[ "$idx" -ge ${#ips[@]} ]]; then
    echo "Worker index $idx out of range (have ${#ips[@]} instances)." >&2
    exit 1
  fi
  echo "${ips[$idx]}"
}

# ssh_hapi [worker_index] [remote command...]
ssh_hapi() {
  local idx=0
  if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    idx="$1"
    shift
  fi
  local ip
  ip="$(ip_for_index "$idx")"
  ssh $(ssh_opts) "${SSH_USER}@${ip}" "$@"
}

rsync_to_index() {
  local idx="$1"
  local ip
  ip="$(ip_for_index "$idx")"
  rsync -az --delete --partial --progress \
    -e "ssh $(ssh_opts)" \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude 'results/' \
    --exclude 'archive/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '*.zip' \
    --exclude 'scripts/aws/.env' \
    --exclude 'scripts/aws/.state' \
    "$REPO_ROOT/" "${SSH_USER}@${ip}:hapi/"
}

current_ssh_cidr() {
  if [[ -n "$SSH_CIDR" ]]; then
    echo "$SSH_CIDR"
    return
  fi
  local ip
  ip="$(curl -fsS --max-time 10 https://checkip.amazonaws.com | tr -d '[:space:]')"
  echo "${ip}/32"
}

account_id() {
  aws sts get-caller-identity --query Account --output text
}

remote_setup_script() {
  cat <<'REMOTE'
set -euo pipefail
cd "$HOME/hapi"
echo "==> GPU"
nvidia-smi
python_bin=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    python_bin="$candidate"
    break
  fi
done
echo "==> Using $python_bin ($("$python_bin" --version))"
if [[ ! -d .venv ]]; then
  "$python_bin" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel
cuda_tag="cu124"
if nvidia-smi >/dev/null 2>&1; then
  driver_cuda="$(nvidia-smi | awk '/CUDA Version/ {print $NF; exit}')"
  case "$driver_cuda" in
    12.1*|12.2*) cuda_tag="cu121" ;;
    12.4*|12.5*) cuda_tag="cu124" ;;
    12.6*|12.8*|12.9*|13.*) cuda_tag="cu128" ;;
  esac
  echo "==> Installing torch from https://download.pytorch.org/whl/${cuda_tag} (driver CUDA $driver_cuda)"
fi
python -m pip install torch --index-url "https://download.pytorch.org/whl/${cuda_tag}"
grep -v '^torch' requirements.txt > /tmp/hapi-requirements-no-torch.txt
python -m pip install -r /tmp/hapi-requirements-no-torch.txt
python - <<'PY'
import torch
info = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}
print(info)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch")
PY
python scripts/20_validate_final_v3.py --data-root data/final_325_nodules_v3
python -m unittest
python -m compileall -q scripts src/hapi
echo "==> Remote setup complete"
REMOTE
}
