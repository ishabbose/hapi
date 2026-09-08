#!/usr/bin/env bash
# Create SSH key + security group and launch a CUDA GPU instance.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

need_cmd aws
need_cmd curl

load_state

if [[ -n "${INSTANCE_ID:-}" ]]; then
  state="$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>/dev/null || echo missing)"
  if [[ "$state" == "running" || "$state" == "pending" || "$state" == "stopped" || "$state" == "stopping" ]]; then
    echo "Reusing existing instance $INSTANCE_ID (state=$state)."
    if [[ "$state" == "stopped" ]]; then
      aws ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
      aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    fi
    refresh_public_ip
    echo "SSH: ssh -i $KEY_PATH $SSH_USER@$PUBLIC_IP"
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

echo "==> Launching $INSTANCE_TYPE"
INSTANCE_ID="$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --block-device-mappings "DeviceName=${root_dev},Ebs={VolumeSize=${VOLUME_GB},VolumeType=gp3,DeleteOnTermination=true}" \
  --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=2" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}},{Key=Project,Value=hapi}]" \
  --count 1 \
  --query 'Instances[0].InstanceId' \
  --output text)"

echo "Instance $INSTANCE_ID launching..."
save_state
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"
refresh_public_ip

echo
echo "Ready."
echo "  Instance: $INSTANCE_ID"
echo "  IP:       $PUBLIC_IP"
echo "  SSH:      ssh -i $KEY_PATH $SSH_USER@$PUBLIC_IP"
echo
echo "Next: scripts/aws/03_sync.sh"
