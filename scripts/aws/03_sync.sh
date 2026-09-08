#!/usr/bin/env bash
# Copy the repo and v3.1 dataset to the GPU instance.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

need_cmd rsync
require_instance
refresh_public_ip

echo "==> rsync to $SSH_USER@$PUBLIC_IP:~/hapi"
rsync -az --delete --partial --progress \
  -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'results/' \
  --exclude 'archive/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude '*.zip' \
  --exclude 'scripts/aws/.env' \
  --exclude 'scripts/aws/.state' \
  "$REPO_ROOT/" "${SSH_USER}@${PUBLIC_IP}:hapi/"

echo "Sync complete. Next: scripts/aws/04_remote_setup.sh"
