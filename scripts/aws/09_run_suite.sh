#!/usr/bin/env bash
# Start the locked suite in the background on the fleet (does not wait).
# Usage: ./scripts/aws/09_run_suite.sh [cheap|parallel9|single8]
# Optional env: SUITE=full|screen
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
take_mode_arg "${1:-}" || true
instance_id_list
load_state
apply_runtime_mode
print_mode_banner

if [[ "$SUITE" != "full" && "$SUITE" != "screen" && "$SUITE" != "confirm" ]]; then
  echo "SUITE must be full, screen, or confirm (got $SUITE)" >&2
  exit 1
fi

start_cheap() {
  ssh_hapi 0 'bash -s' <<REMOTE
set -euo pipefail
cd "\$HOME/hapi"
source .venv/bin/activate
mkdir -p logs aws_done
rm -f aws_done/worker
nohup bash -c 'python scripts/23_run_ablation_suite.py --suite ${SUITE} --execute --device cuda --workers ${DATALOADER_WORKERS} --evaluation-bootstrap ${EVALUATION_BOOTSTRAP} --aggregate-bootstrap ${AGGREGATE_BOOTSTRAP} && touch aws_done/worker' > logs/suite.log 2>&1 &
echo \$! > logs/suite.pid
echo "cheap suite started, pid=\$(cat logs/suite.pid)"
REMOTE
}

start_parallel9() {
  local i
  for i in $(seq 0 8); do
    local variant="${ALL_VARIANTS[$i]}"
    echo "==> start worker $i variant=$variant"
    ssh_hapi "$i" 'bash -s' <<REMOTE
set -euo pipefail
cd "\$HOME/hapi"
source .venv/bin/activate
mkdir -p logs aws_done
rm -f aws_done/worker
nohup bash -c 'python scripts/23_run_ablation_suite.py --suite ${SUITE} --variants ${variant} --execute --device cuda --workers ${DATALOADER_WORKERS} --evaluation-bootstrap ${EVALUATION_BOOTSTRAP} --aggregate-bootstrap ${AGGREGATE_BOOTSTRAP} --skip-aggregate --skip-figures && touch aws_done/worker' > logs/suite.log 2>&1 &
echo \$! > logs/suite.pid
echo "worker $i started, pid=\$(cat logs/suite.pid)"
REMOTE
  done
}

start_single8() {
  local groups_csv
  groups_csv="$(IFS='|'; echo "${SINGLE8_VARIANT_GROUPS[*]}")"
  ssh_hapi 0 'bash -s' <<REMOTE
set -euo pipefail
cd "\$HOME/hapi"
source .venv/bin/activate
mkdir -p logs aws_done
rm -f aws_done/gpu_*
IFS='|' read -r -a groups <<< "${groups_csv}"
gpu=0
for variants in "\${groups[@]}"; do
  rm -f aws_done/gpu_\${gpu}
  nohup bash -c "CUDA_VISIBLE_DEVICES=\${gpu} python scripts/23_run_ablation_suite.py --suite ${SUITE} --variants \${variants} --execute --device cuda --workers ${DATALOADER_WORKERS} --evaluation-bootstrap ${EVALUATION_BOOTSTRAP} --aggregate-bootstrap ${AGGREGATE_BOOTSTRAP} --skip-aggregate --skip-figures && touch aws_done/gpu_\${gpu}" > logs/gpu_\${gpu}.log 2>&1 &
  echo \$! > logs/gpu_\${gpu}.pid
  echo "GPU \${gpu} variants=\${variants} pid=\$(cat logs/gpu_\${gpu}.pid)"
  gpu=\$((gpu + 1))
done
REMOTE
}

case "$MODE" in
  cheap) start_cheap ;;
  parallel9) start_parallel9 ;;
  single8) start_single8 ;;
esac

echo
echo "Jobs are running in the background (nohup)."
echo "  Logs: ./scripts/aws/ssh.sh 0   then  tail -f ~/hapi/logs/suite.log"
echo "  Wait + aggregate: ./scripts/aws/10_wait_and_aggregate.sh"
echo "Stop billing when finished: ./scripts/aws/06_stop.sh"
