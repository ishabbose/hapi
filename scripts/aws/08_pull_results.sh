#!/usr/bin/env bash
# Copy results/ from every GPU instance back to this laptop (merged).
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

need_cmd rsync
instance_id_list
mkdir -p "$REPO_ROOT/results"
idx=0
for _id in "${INSTANCE_ID_ARR[@]}"; do
  ip="$(ip_for_index "$idx")"
  echo "==> pull worker $idx ($ip)"
  rsync -az --partial --progress \
    -e "ssh $(ssh_opts)" \
    "${SSH_USER}@${ip}:hapi/results/" "$REPO_ROOT/results/"
  idx=$((idx + 1))
done
echo "Results merged into $REPO_ROOT/results"
