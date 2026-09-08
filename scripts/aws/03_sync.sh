#!/usr/bin/env bash
# Copy the repo and v3.1 dataset to every GPU instance in the fleet.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

need_cmd rsync
instance_id_list
print_mode_banner

idx=0
for _id in "${INSTANCE_ID_ARR[@]}"; do
  echo "==> rsync worker $idx"
  rsync_to_index "$idx"
  idx=$((idx + 1))
done

echo "Sync complete. Next: ./scripts/aws/04_remote_setup.sh"
