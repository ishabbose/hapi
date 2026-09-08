#!/usr/bin/env bash
# Create a CUDA venv on the instance and validate the v3.1 dataset.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

require_instance

ssh_hapi 'bash -s' <<'REMOTE'
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
# shellcheck disable=SC1091
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
echo
echo "Next: scripts/aws/05_run_smoke.sh"
echo "Then, on the instance or via ssh:"
echo "  python scripts/23_run_ablation_suite.py --suite screen --execute --device cuda --workers 4 --evaluation-bootstrap 2000 --aggregate-bootstrap 10000"
