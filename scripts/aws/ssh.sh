#!/usr/bin/env bash
# Interactive SSH. Usage: ./scripts/aws/ssh.sh [worker_index]
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
instance_id_list
idx=0
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  idx="$1"
  shift
fi
ip="$(ip_for_index "$idx")"
echo "Connecting to worker $idx ($ip)"
exec ssh $(ssh_opts) "${SSH_USER}@${ip}" "$@"
