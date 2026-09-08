# AWS GPU setup (almost no console)

These scripts launch a CUDA instance, copy this repo plus `data/final_325_nodules_v3`, install a GPU PyTorch venv, and run smoke. Run every script **on your laptop** from the repo root, except where noted.

Default instance: `g4dn.xlarge` (1× NVIDIA T4, 16 GB). Change `INSTANCE_TYPE` in `scripts/aws/.env` if you want `g5.xlarge` (A10G).

## Console steps you cannot skip

Do these **once** per AWS account. Everything after that is CLI.

1. Create an AWS account and turn on MFA on the root user.
2. Create an IAM user (or Identity Center user) with permission to use EC2, Service Quotas, and optionally Budgets. Create an **access key**.
3. If you have never launched a GPU VM, G-instance **quota is often 0**. The check script will try to request 8 G/VT vCPUs. Approval can take hours; if the API is denied, open Service Quotas in the console for “Running On-Demand G and VT instances” and request at least 4 vCPUs.

Never paste access keys into this repo.

## Laptop one-time

```bash
# macOS
brew install awscli

aws configure    # paste access key, secret, region us-east-1, output json

cp scripts/aws/config.env.example scripts/aws/.env
# edit scripts/aws/.env if you want a different region or instance type
```

## Every training session

```bash
chmod +x scripts/aws/*.sh

./scripts/aws/01_check_local.sh
./scripts/aws/02_launch_gpu.sh
./scripts/aws/03_sync.sh
./scripts/aws/04_remote_setup.sh
./scripts/aws/05_run_smoke.sh
```

Open a shell on the instance:

```bash
./scripts/aws/ssh.sh
cd ~/hapi
source .venv/bin/activate
```

Screen suite (long; leave the SSH session running or use `tmux`):

```bash
tmux new -s hapi
python scripts/23_run_ablation_suite.py --suite screen --execute --device cuda --workers 4 --evaluation-bootstrap 2000 --aggregate-bootstrap 10000
```

Copy results back to the laptop when a run finishes:

```bash
./scripts/aws/08_pull_results.sh
```

## Money

GPU instances bill while **running**. Stop when you are not training:

```bash
./scripts/aws/06_stop.sh     # keeps the disk; 02_launch_gpu.sh starts it again
./scripts/aws/07_terminate.sh  # deletes the VM; next 02 launches a new one
```

`01_check_local.sh` tries to create a monthly cost budget named `hapi-monthly-cap` (default $25). Attach an email in Budgets if you want a mail warning.

Do not run two suite processes against the same `results/final_agreement/...` folders (laptop CPU + AWS GPU).

## If launch fails

| Error | What to do |
|---|---|
| `VcpuLimitExceeded` / `InsufficientInstanceCapacity` | Quota not approved yet, or that size is sold out — retry later or change `INSTANCE_TYPE` / region |
| `UnauthorizedOperation` | IAM user needs EC2 + Service Quotas |
| SSH timeout | Your public IP changed; rerun `02_launch_gpu.sh` (it reuses the instance) after updating `SSH_CIDR` in `.env` |
| PyTorch `cuda False` | Rerun `04_remote_setup.sh`; confirm `nvidia-smi` inside `./scripts/aws/ssh.sh` |

State is stored in gitignored `scripts/aws/.state`. The private key is `~/.ssh/hapi-aws-<region>.pem` and is **not** recoverable from AWS if you delete it.
