#!/usr/bin/env bash
# Copy results/ from the GPU instance back to this laptop.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

need_cmd rsync
require_instance
refresh_public_ip
mkdir -p "$REPO_ROOT/results"
rsync -az --partial --progress \
  -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes" \
  "${SSH_USER}@${PUBLIC_IP}:hapi/results/" "$REPO_ROOT/results/"
echo "Results copied to $REPO_ROOT/results"
