#!/usr/bin/env bash
# Interactive SSH into the GPU instance (repo is ~/hapi).
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
require_instance
refresh_public_ip
exec ssh -i "$KEY_PATH" \
  -o StrictHostKeyChecking=accept-new \
  -o IdentitiesOnly=yes \
  -o ServerAliveInterval=30 \
  "${SSH_USER}@${PUBLIC_IP}" "$@"
