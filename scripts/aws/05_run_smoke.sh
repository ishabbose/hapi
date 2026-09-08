#!/usr/bin/env bash
# Dry-run then execute the smoke suite on the GPU instance.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

require_instance

ssh_hapi 'bash -s' <<'REMOTE'
set -euo pipefail
cd "$HOME/hapi"
source .venv/bin/activate
python scripts/23_run_ablation_suite.py --suite smoke
python scripts/23_run_ablation_suite.py --suite smoke --execute --device cuda --workers 4 --evaluation-bootstrap 20
REMOTE
