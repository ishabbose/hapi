# AWS GPU setup (almost no console)

These scripts launch CUDA hardware, copy this repo plus `data/final_325_nodules_v3`, install a GPU PyTorch venv, and run smoke or the locked suites. Run every numbered script **on your laptop** from the repo root.

## Pick a mode

Set `MODE` in `scripts/aws/.env`, or pass it as the first argument: `./scripts/aws/02_launch_gpu.sh parallel9`.

| Mode | Hardware | Full-suite wall clock (guess) | Full-suite $ (guess, on-demand us-east-1) | Use when |
|---|---|---|---|---|
| `cheap` | 1× `g4dn.xlarge` (T4) ~$0.53/h | ~2–3 weeks | ~$150–$350 | Tight budget, one GPU quota |
| `parallel9` | 9× `g5.xlarge` (A10G) ~$9.05/h | ~1.5–3 days | ~$200–$400 | Fastest finish; need ~36 G/VT vCPUs |
| `single8` | 1× `g5.48xlarge` (8× A10G) ~$16.29/h | ~2–3 days | ~$500–$800 | One machine; need ~192 G/VT vCPUs |

`single8` can use A100s with `SINGLE8_INSTANCE_TYPE=p4d.24xlarge` (separate P-instance quota). Times are estimates until you time one fold.

Set `SUITE=screen` in `.env` for the 45-run seed-42 matrix instead of `full` (135 runs).

## Console steps you cannot skip

Do these **once** per AWS account.

1. Create an AWS account and turn on MFA on the root user.
2. Create an IAM user (or Identity Center user) with EC2, Service Quotas, and optionally Budgets. Create an **access key**.
3. G-instance **quota is often 0**. `01_check_local.sh` requests vCPUs for the selected mode (`cheap` 8, `parallel9` 40, `single8` 192). Approval can take hours.

Never paste access keys into this repo.

## Laptop one-time

```bash
brew install awscli
aws configure
cp scripts/aws/config.env.example scripts/aws/.env
# edit MODE, SUITE, BUDGET_LIMIT
chmod +x scripts/aws/*.sh
```

## Every training session

```bash
./scripts/aws/01_check_local.sh cheap          # or parallel9 / single8
./scripts/aws/02_launch_gpu.sh cheap
./scripts/aws/03_sync.sh
./scripts/aws/04_remote_setup.sh
./scripts/aws/05_run_smoke.sh                  # GPU 0 only
SUITE=full ./scripts/aws/09_run_suite.sh      # background nohup
./scripts/aws/10_wait_and_aggregate.sh        # poll, merge, aggregate
./scripts/aws/08_pull_results.sh
./scripts/aws/06_stop.sh                      # stop billing for compute
```

SSH to a worker (0–8 in `parallel9`):

```bash
./scripts/aws/ssh.sh 0
tail -f ~/hapi/logs/suite.log
```

Do not run the same suite on this laptop and AWS at the same time.

## Money

GPU instances bill while **running**. `cheap` left on is ~$13/day; `parallel9` ~$217/day; `single8` ~$391/day.

```bash
./scripts/aws/06_stop.sh       # keeps disks
./scripts/aws/07_terminate.sh # deletes the fleet
```

Raise `BUDGET_LIMIT` before `parallel9` or `single8`. Changing `MODE` after a launch requires `07_terminate.sh` first.

## If launch fails

| Error | What to do |
|---|---|
| `VcpuLimitExceeded` | Quota too low for this MODE — wait for the request, or use `cheap` |
| `InsufficientInstanceCapacity` | That size is sold out — retry later or another region |
| SSH timeout | Your IP changed; set `SSH_CIDR` in `.env` and rerun `02_launch_gpu.sh` |
| PyTorch `cuda False` | Rerun `04_remote_setup.sh`; `nvidia-smi` via `./scripts/aws/ssh.sh 0` |

State is gitignored in `scripts/aws/.state`. The private key is `~/.ssh/hapi-aws-<region>.pem` and cannot be re-downloaded from AWS.
