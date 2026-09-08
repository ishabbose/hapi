#!/usr/bin/env bash
# Create a CUDA venv on every instance and validate the v3.1 dataset.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

instance_id_list
print_mode_banner

idx=0
for _id in "${INSTANCE_ID_ARR[@]}"; do
  echo "==> setup worker $idx"
  ssh_hapi "$idx" 'bash -s' < <(remote_setup_script)
  idx=$((idx + 1))
done

echo
echo "Next: ./scripts/aws/05_run_smoke.sh"
echo "Full/screen suite: ./scripts/aws/09_run_suite.sh"
