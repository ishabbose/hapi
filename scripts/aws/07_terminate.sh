#!/usr/bin/env bash
# Permanently delete the GPU fleet. Does not delete the SSH key pair.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

instance_id_list
echo "Terminating ${INSTANCE_ID_ARR[*]} ..."
aws ec2 terminate-instances --instance-ids "${INSTANCE_ID_ARR[@]}" >/dev/null
aws ec2 wait instance-terminated --instance-ids "${INSTANCE_ID_ARR[@]}"
INSTANCE_IDS=""
PUBLIC_IPS=""
INSTANCE_ID=""
PUBLIC_IP=""
save_state
echo "Terminated. The security group $SG_NAME was kept so the next launch is faster."
