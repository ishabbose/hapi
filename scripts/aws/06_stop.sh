#!/usr/bin/env bash
# Stop every GPU instance in the fleet (keeps disks; compute billing stops).
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

instance_id_list
echo "Stopping ${INSTANCE_ID_ARR[*]} ..."
aws ec2 stop-instances --instance-ids "${INSTANCE_ID_ARR[@]}" >/dev/null
aws ec2 wait instance-stopped --instance-ids "${INSTANCE_ID_ARR[@]}"
echo "Stopped. Disk still incurs a small EBS charge. Start later with ./scripts/aws/02_launch_gpu.sh"
