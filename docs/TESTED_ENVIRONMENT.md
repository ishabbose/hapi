# Tested integration environment

The dataset validator, model/loss checks, and isolated smoke train/evaluation
runs were executed with the following environment on 2026-09-04:

| Component | Version |
|---|---:|
| Python | 3.12.13 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |
| PyTorch | 2.14.0+cpu |
| Platform | Linux x86_64, glibc 2.39 |

These checks establish interface and data-integrity compatibility. Final model
training is configured for CUDA automatic mixed precision; each final run's
`config.json` records the actual PyTorch version, resolved device, AMP state,
dataset fingerprint, manifest hash, hyperparameters, and parameter count.

The released fold assignment is frozen by
`final_325_manifest.csv` (SHA-256
`0d0e7e5950cf71371e7ac4532aa80ab44a4b3e9b303d570ab646a7e9fec9d06a`).
Use the shipped manifest rather than regenerating folds for a published run.
