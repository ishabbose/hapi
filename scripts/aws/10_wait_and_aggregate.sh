#!/usr/bin/env bash
# Poll workers until done, merge results onto worker 0, then aggregate.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
instance_id_list
load_state
apply_runtime_mode
print_mode_banner

worker_done() {
  local idx="$1"
  ssh_hapi "$idx" 'test -f "$HOME/hapi/aws_done/worker"' >/dev/null 2>&1
}

single8_done() {
  ssh_hapi 0 'bash -s' <<'REMOTE'
set -euo pipefail
for g in 0 1 2 3 4 5 6 7; do
  test -f "$HOME/hapi/aws_done/gpu_$g" || exit 1
done
REMOTE
}

echo "Polling every 120s until MODE=$MODE workers finish..."
while true; do
  ready=1
  case "$MODE" in
    cheap)
      worker_done 0 || ready=0
      ;;
    parallel9)
      for i in $(seq 0 8); do
        worker_done "$i" || ready=0
      done
      ;;
    single8)
      single8_done || ready=0
      ;;
  esac
  if [[ "$ready" -eq 1 ]]; then
    echo "All workers reported done."
    break
  fi
  echo "$(date -u +%H:%M:%SZ) still running..."
  sleep 120
done

if [[ "$MODE" == "parallel9" ]]; then
  echo "==> Merge results onto worker 0"
  for i in $(seq 1 8); do
    src_ip="$(ip_for_index "$i")"
    dst_ip="$(ip_for_index 0)"
    echo "    worker $i -> 0"
    ssh $(ssh_opts) "${SSH_USER}@${dst_ip}" "mkdir -p \$HOME/hapi/results"
    rsync -az --partial \
      -e "ssh $(ssh_opts)" \
      "${SSH_USER}@${src_ip}:hapi/results/" \
      "${SSH_USER}@${dst_ip}:hapi/results/"
  done
fi

echo "==> Aggregate on worker 0"
ssh_hapi 0 'bash -s' <<REMOTE
set -euo pipefail
cd "\$HOME/hapi"
source .venv/bin/activate
python scripts/23_run_ablation_suite.py --suite ${SUITE} --execute --device cpu --workers 0 --evaluation-bootstrap ${EVALUATION_BOOTSTRAP} --aggregate-bootstrap ${AGGREGATE_BOOTSTRAP}
REMOTE

echo "Aggregation finished. Pull with ./scripts/aws/08_pull_results.sh then ./scripts/aws/06_stop.sh"
