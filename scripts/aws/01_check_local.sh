#!/usr/bin/env bash
# Check local tools, AWS credentials, GPU quota for the selected MODE, and a spending alarm.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
take_mode_arg "${1:-}" || true

need_cmd aws
need_cmd ssh
need_cmd rsync
need_cmd curl
need_cmd python3

print_mode_banner
echo
echo "Modes (pick with MODE= in scripts/aws/.env or pass cheap|parallel9|single8):"
echo "  cheap      1x g4dn.xlarge     ~\$0.53/h   sequential full suite (slowest, cheapest hourly)"
echo "  parallel9  9x g5.xlarge       ~\$9.05/h   fastest wall-clock (one variant per VM)"
echo "  single8    1x g5.48xlarge     ~\$16.29/h  8 GPUs on one box"
echo

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

QUOTA_CODE="L-DB2E81BA"
echo "==> GPU On-Demand G/VT vCPU quota ($QUOTA_CODE); this mode needs >= $MIN_VCPU_QUOTA"
quota_value="$(aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code "$QUOTA_CODE" \
  --query 'Quota.Value' \
  --output text 2>/dev/null || echo "")"

if [[ -z "$quota_value" || "$quota_value" == "None" ]]; then
  echo "Could not read quota (IAM may lack service-quotas). Launch may still work."
elif awk "BEGIN {exit !($quota_value < $MIN_VCPU_QUOTA)}"; then
  echo "Current G/VT On-Demand vCPU quota is $quota_value (need at least $MIN_VCPU_QUOTA for MODE=$MODE)."
  if [[ "$REQUEST_GPU_QUOTA" == "1" ]]; then
    echo "Requesting quota increase to $QUOTA_REQUEST vCPUs..."
    aws service-quotas request-service-quota-increase \
      --service-code ec2 \
      --quota-code "$QUOTA_CODE" \
      --desired-value "$QUOTA_REQUEST" \
      >/dev/null && echo "Quota request submitted. AWS may take hours and can email you."
  else
    echo "Set REQUEST_GPU_QUOTA=1 in scripts/aws/.env or request it in Service Quotas."
  fi
else
  echo "Quota is $quota_value vCPUs — enough for MODE=$MODE."
fi

echo "==> Monthly budget alarm (\$$BUDGET_LIMIT)"
acct="$(account_id)"
if aws budgets describe-budget --account-id "$acct" --budget-name hapi-monthly-cap >/dev/null 2>&1; then
  echo "Budget hapi-monthly-cap already exists."
else
  tmp="$(mktemp)"
  cat >"$tmp" <<EOF
{
  "BudgetName": "hapi-monthly-cap",
  "BudgetLimit": {"Amount": "${BUDGET_LIMIT}", "Unit": "USD"},
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
EOF
  if aws budgets create-budget --account-id "$acct" --budget "file://$tmp" >/dev/null 2>&1; then
    echo "Created cost budget hapi-monthly-cap at \$$BUDGET_LIMIT / month."
    echo "Add an email subscriber in Budgets if you want a mail alert (one console visit)."
  else
    echo "Could not create a budget (missing budgets permission). Watch the billing dashboard."
  fi
  rm -f "$tmp"
fi

echo
echo "Local checks passed. Next: ./scripts/aws/02_launch_gpu.sh $MODE"
