#!/usr/bin/env bash
# Permanently delete the GPU instance. Does not delete the SSH key pair.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

require_instance
echo "Terminating $INSTANCE_ID ..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null
aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
INSTANCE_ID=""
PUBLIC_IP=""
save_state
echo "Terminated. The security group $SG_NAME was kept so the next launch is faster."
