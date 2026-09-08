#!/usr/bin/env bash
# Stop the GPU instance (keeps the disk; billing for compute stops).
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

require_instance
echo "Stopping $INSTANCE_ID ..."
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" >/dev/null
aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
echo "Stopped. Disk still incurs a small EBS charge. Start later with scripts/aws/02_launch_gpu.sh"
