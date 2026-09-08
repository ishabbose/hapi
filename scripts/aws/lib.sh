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
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
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

export AWS_DEFAULT_REGION="$AWS_REGION"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1" >&2
    exit 1
  }
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
INSTANCE_ID=${INSTANCE_ID:-}
PUBLIC_IP=${PUBLIC_IP:-}
SG_ID=${SG_ID:-}
KEY_PATH=$(printf '%q' "$KEY_PATH")
KEY_NAME=$(printf '%q' "$KEY_NAME")
AWS_REGION=$(printf '%q' "$AWS_REGION")
SSH_USER=$(printf '%q' "$SSH_USER")
EOF
}

require_instance() {
  load_state
  if [[ -z "${INSTANCE_ID:-}" ]]; then
    echo "No instance in $STATE_FILE. Run scripts/aws/02_launch_gpu.sh first." >&2
    exit 1
  fi
}

refresh_public_ip() {
  require_instance
  PUBLIC_IP="$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)"
  if [[ -z "$PUBLIC_IP" || "$PUBLIC_IP" == "None" ]]; then
    echo "Instance $INSTANCE_ID has no public IP (is it stopped?)." >&2
    exit 1
  fi
  save_state
}

ssh_hapi() {
  refresh_public_ip
  ssh -i "$KEY_PATH" \
    -o StrictHostKeyChecking=accept-new \
    -o IdentitiesOnly=yes \
    -o ServerAliveInterval=30 \
    "${SSH_USER}@${PUBLIC_IP}" "$@"
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
