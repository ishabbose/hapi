#!/usr/bin/env bash
# Dry-run then execute the smoke suite on worker 0 / GPU 0.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

require_instance
print_mode_banner

ssh_hapi 0 'bash -s' <<REMOTE
set -euo pipefail
cd "\$HOME/hapi"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python scripts/23_run_ablation_suite.py --suite smoke
python scripts/23_run_ablation_suite.py --suite smoke --execute --device cuda --workers ${DATALOADER_WORKERS} --evaluation-bootstrap 20
REMOTE
