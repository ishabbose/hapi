#!/usr/bin/env bash
# Create SSH key + security group and launch GPU instances for MODE.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
take_mode_arg "${1:-}" || true

need_cmd aws
need_cmd curl

WANTED_MODE="$MODE"
print_mode_banner

load_state
SAVED_MODE="${MODE:-}"
SAVED_IDS="${INSTANCE_IDS:-${INSTANCE_ID:-}}"
MODE="$WANTED_MODE"
apply_runtime_mode

if [[ -n "$SAVED_IDS" ]]; then
  # shellcheck disable=SC2206
  check_ids=($SAVED_IDS)
  first_state="$(aws ec2 describe-instances \
    --instance-ids "${check_ids[0]}" \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>/dev/null || echo missing)"
  if [[ "$first_state" != "missing" && "$first_state" != "terminated" && "$first_state" != "shutting-down" ]]; then
    if [[ -n "$SAVED_MODE" && "$SAVED_MODE" != "$WANTED_MODE" ]]; then
      echo "Existing fleet is MODE=$SAVED_MODE ($SAVED_IDS)." >&2
      echo "Terminate it with ./scripts/aws/07_terminate.sh before launching MODE=$WANTED_MODE." >&2
      exit 1
    fi
    echo "Reusing ${#check_ids[@]} instance(s) (state=$first_state)."
    INSTANCE_IDS="$SAVED_IDS"
    INSTANCE_ID="${check_ids[0]}"
    running=()
    stopped=()
    for id in "${check_ids[@]}"; do
      st="$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)"
      case "$st" in
        stopped|stopping) stopped+=("$id") ;;
        running|pending) running+=("$id") ;;
        *) echo "Instance $id is $st; will not reuse." >&2; exit 1 ;;
      esac
    done
    if [[ ${#stopped[@]} -gt 0 ]]; then
      aws ec2 start-instances --instance-ids "${stopped[@]}" >/dev/null
    fi
    aws ec2 wait instance-running --instance-ids "${check_ids[@]}"
    refresh_public_ips
    echo "SSH worker 0: ./scripts/aws/ssh.sh 0"
    exit 0
  fi
fi

mkdir -p "$(dirname "$KEY_PATH")"
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
  echo "==> Creating key pair $KEY_NAME"
  aws ec2 create-key-pair \
    --key-name "$KEY_NAME" \
    --query 'KeyMaterial' \
    --output text >"$KEY_PATH"
  chmod 400 "$KEY_PATH"
else
  echo "==> Key pair $KEY_NAME already exists in AWS."
  if [[ ! -f "$KEY_PATH" ]]; then
    echo "The private key is not at $KEY_PATH. AWS cannot re-download it." >&2
    echo "Set KEY_PATH in scripts/aws/.env to your existing .pem, or delete the key pair and rerun." >&2
    exit 1
  fi
fi

cidr="$(current_ssh_cidr)"
echo "==> SSH ingress $cidr"

vpc_id="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
if [[ -z "$vpc_id" || "$vpc_id" == "None" ]]; then
  echo "No default VPC in $AWS_REGION. Create a default VPC (rare one-time console/API step)." >&2
  exit 1
fi

SG_ID="$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$vpc_id" \
  --query 'SecurityGroups[0].GroupId' \
  --output text 2>/dev/null || echo "")"
if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
  echo "==> Creating security group $SG_NAME"
  SG_ID="$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "SSH for HAPI GPU instance" \
    --vpc-id "$vpc_id" \
    --query GroupId --output text)"
fi
if ! aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr "$cidr" >/dev/null 2>&1; then
  echo "SSH rule for $cidr already present or not needed."
fi

if [[ -z "$AMI_ID" ]]; then
  echo "==> Resolving Deep Learning PyTorch GPU AMI"
  AMI_ID="$(aws ec2 describe-images \
    --owners amazon \
    --filters \
      "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch *Ubuntu*" \
      "Name=state,Values=available" \
      "Name=architecture,Values=x86_64" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)"
  if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    AMI_ID="$(aws ec2 describe-images \
      --owners amazon \
      --filters \
        "Name=name,Values=Deep Learning NVIDIA GPU AMI *Ubuntu*" \
        "Name=state,Values=available" \
        "Name=architecture,Values=x86_64" \
      --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
      --output text)"
  fi
fi
if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
  echo "Could not auto-select an AMI. Set AMI_ID in scripts/aws/.env" >&2
  exit 1
fi
ami_name="$(aws ec2 describe-images --image-ids "$AMI_ID" --query 'Images[0].Name' --output text)"
root_dev="$(aws ec2 describe-images --image-ids "$AMI_ID" --query 'Images[0].RootDeviceName' --output text)"
echo "==> AMI $AMI_ID ($ami_name)"

echo "==> Launching $INSTANCE_COUNT x $INSTANCE_TYPE"
id_blob="$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --block-device-mappings "DeviceName=${root_dev},Ebs={VolumeSize=${VOLUME_GB},VolumeType=gp3,DeleteOnTermination=true}" \
  --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=2" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}-${MODE}},{Key=Project,Value=hapi},{Key=HapiMode,Value=${MODE}}]" \
  --count "$INSTANCE_COUNT" \
  --query 'Instances[*].InstanceId' \
  --output text)"
# shellcheck disable=SC2206
launched=($id_blob)

if [[ ${#launched[@]} -ne $INSTANCE_COUNT ]]; then
  echo "Expected $INSTANCE_COUNT instance ids, got: ${launched[*]:-none}" >&2
  exit 1
fi

INSTANCE_IDS="${launched[*]}"
INSTANCE_ID="${launched[0]}"
save_state

index=0
for id in "${launched[@]}"; do
  aws ec2 create-tags --resources "$id" --tags "Key=Name,Value=${NAME}-${MODE}-${index}" >/dev/null
  index=$((index + 1))
done

echo "Waiting for ${#launched[@]} instance(s)..."
aws ec2 wait instance-running --instance-ids "${launched[@]}"
aws ec2 wait instance-status-ok --instance-ids "${launched[@]}"
refresh_public_ips

echo
echo "Ready (MODE=$MODE)."
echo "  Instances: $INSTANCE_IDS"
echo "  IPs:       $PUBLIC_IPS"
echo "  SSH:       ./scripts/aws/ssh.sh 0"
echo
echo "Next: ./scripts/aws/03_sync.sh"
