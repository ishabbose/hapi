#!/usr/bin/env bash
# Check local tools, AWS credentials, GPU quota, and a spending alarm.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

need_cmd aws
need_cmd ssh
need_cmd rsync
need_cmd curl
need_cmd python3

echo "==> AWS identity"
aws sts get-caller-identity

echo "==> Region $AWS_REGION"
if ! aws ec2 describe-availability-zones --region "$AWS_REGION" >/dev/null; then
  echo "Cannot call EC2 in $AWS_REGION." >&2
  exit 1
fi

DATA_ROOT="$REPO_ROOT/data/final_325_nodules_v3"
if [[ ! -f "$DATA_ROOT/final_325_manifest.csv" ]]; then
  echo "Missing $DATA_ROOT/final_325_manifest.csv" >&2
  echo "The v3.1 dataset must exist on this laptop before sync." >&2
  exit 1
fi
echo "==> Local v3.1 dataset found"

# G and VT On-Demand vCPUs (needed for g4dn/g5).
QUOTA_CODE="L-DB2E81BA"
echo "==> GPU On-Demand quota ($QUOTA_CODE)"
quota_value="$(aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code "$QUOTA_CODE" \
  --query 'Quota.Value' \
  --output text 2>/dev/null || echo "")"

if [[ -z "$quota_value" || "$quota_value" == "None" ]]; then
  echo "Could not read quota (IAM may lack service-quotas). Launch may still work."
elif awk "BEGIN {exit !($quota_value < 4)}"; then
  echo "Current G/VT On-Demand vCPU quota is $quota_value (need at least 4 for $INSTANCE_TYPE)."
  if [[ "$REQUEST_GPU_QUOTA" == "1" ]]; then
    echo "Requesting quota increase to 8 vCPUs..."
    aws service-quotas request-service-quota-increase \
      --service-code ec2 \
      --quota-code "$QUOTA_CODE" \
      --desired-value 8 \
      >/dev/null && echo "Quota request submitted. AWS may take hours and can email you."
  else
    echo "Set REQUEST_GPU_QUOTA=1 in scripts/aws/.env or request it in Service Quotas."
  fi
else
  echo "Quota is $quota_value vCPUs — enough to launch $INSTANCE_TYPE."
fi

echo "==> Monthly budget alarm (\$$BUDGET_LIMIT)"
acct="$(account_id)"
if aws budgets describe-budget --account-id "$acct" --budget-name hapi-monthly-cap >/dev/null 2>&1; then
  echo "Budget hapi-monthly-cap already exists."
else
  email="$(aws sts get-caller-identity --query Arn --output text)"
  tmp="$(mktemp)"
  cat >"$tmp" <<EOF
{
  "BudgetName": "hapi-monthly-cap",
  "BudgetLimit": {"Amount": "${BUDGET_LIMIT}", "Unit": "USD"},
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
EOF
  notify="$(mktemp)"
  cat >"$notify" <<EOF
[
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": []
  }
]
EOF
  if aws budgets create-budget --account-id "$acct" --budget "file://$tmp" >/dev/null 2>&1; then
    echo "Created cost budget hapi-monthly-cap at \$$BUDGET_LIMIT / month."
    echo "Add an email subscriber in Budgets if you want a mail alert (one console visit)."
  else
    echo "Could not create a budget (missing budgets permission). Watch the billing dashboard."
  fi
  rm -f "$tmp" "$notify"
fi

echo
echo "Local checks passed. Next: scripts/aws/02_launch_gpu.sh"
