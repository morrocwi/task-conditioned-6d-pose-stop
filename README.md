# Task-Conditioned Early Stopping for Iterative 6D Pose Refinement

Public, implementation-first research repository.

> **Question:** can pose refinement stop because the downstream task is already ready, even when the estimator itself would continue refining?

The central event is:

```text
k_task < k_estimator
```

where `k_task` is the first stage licensed by the downstream task gate and `k_estimator` is the estimator-side convergence stage.

## What is now executable

The repository contains two evidence layers:

1. **Finite diagnostic** — exhaustive toy routing fixture (`benchmark.py`).
2. **Numerical 6D backend** — real iterative point-to-point ICP/Kabsch registration on generated 3-D point clouds (`experiments/numerical_icp.py`).

The numerical experiment uses disjoint **tuning**, **safety-calibration**, and **held-out test** splits. Hidden ground-truth pose error is used only for calibration/evaluation and is never passed to the task gate.

It compares the same refinement trajectories under:

```text
fixed-8 | full | estimator-stop | task-stop
```

for three downstream predicates:

```text
top suction | label alignment | keyed insertion
```

## Standard-profile result

Held-out in-distribution test: 120 episodes; thresholds tuned on 96 separate episodes and checked on another 96 safety-calibration episodes.

| Task | estimator mean k | task mean k | k_task < k_estimator | estimator completion | task completion | task HOLD |
|---|---:|---:|---:|---:|---:|---:|
| top suction | 14.033 | 10.292 | 92.50% | 94.17% | 93.33% | 6.67% |
| label alignment | 14.033 | 10.550 | 90.83% | 93.33% | 91.67% | 6.67% |
| keyed insertion | 14.033 | 11.542 | 90.00% | 92.50% | 90.83% | 9.17% |

Paired bootstrap intervals for `k_task - k_estimator` are strictly below zero for all three in-distribution tasks in the frozen standard run. See [`results/NUMERICAL_RESULTS.md`](results/NUMERICAL_RESULTS.md).

This is **not** a real RGB-D/GPU/robot benchmark. Wall-clock time is a machine-specific secondary readout.

## Failure boundary is part of the result

The same frozen gate is also tested under a declared `2x` sensor-noise shift. Top suction remains usable in this fixture, but label alignment loses substantial completion through HOLD and keyed insertion returns HOLD for every stress episode.

That negative result is deliberate: the repository does not hide distribution-shift failure behind success-conditional-on-ACT metrics.

## Run it yourself

### Fastest route

Requires Python 3.10+.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python reproduce.py --profile quick
```

Outputs are written to:

```text
artifacts/numerical/
```

For the frozen larger experiment:

```bash
python reproduce.py --profile standard
```

On a typical laptop the numerical standard profile is intended to finish in tens of seconds, not hours; timing depends on hardware.

## Automatic checks

GitHub Actions independently reruns the numerical fixture on Linux, macOS, and Windows and uploads result artifacts. Tests include exact Kabsch recovery, pose identity, task switching, a coupled insertion counterexample, disjoint split seeds, no hidden-ground-truth argument in the gate interface, evidence-boundary checks, and glosa claim-card disclosure checks.

## Evidence discipline

This repository applies the five-question / E-A-D discipline from **glosa — Rigour Without Infrastructure** to the research record:

```text
Q1 what was actually seen/run?
Q2 what does the record alone separate?
Q3 what did AI add?
Q4 what is assumed?
Q5 what independent check was run?
```

See [`evidence/claim_card.json`](evidence/claim_card.json).

Current evidence ceiling: **K1 public provisional / I4 mechanical-original-record checks**. Independent external human replication and real robotic validation have not yet occurred.

## Claim boundary

Supported now:

- the public task-stop implementation runs on iterative numerical 6D registration;
- on the frozen in-distribution numerical fixture, task-conditioned stopping often occurs earlier than the estimator criterion;
- ACT, HOLD, unsafe ACT, completion, iterations, and timing are all reported;
- distribution shift can erase the benefit or force HOLD.

Still open / HOLD:

- real RGB-D or GPU acceleration;
- FoundationPose-specific speedup;
- physical manipulation non-inferiority;
- safety certification;
- IDM-specific advantage.

## Why IDM is not the headline

IDM remains an optional finite routing backend in the older rich-action diagnostic. It is not the novelty claim. The present research question is backend-independent: **estimator convergence is not the same condition as downstream task sufficiency.**

## Author

Yaoharee Lahtee  
Open Civil Science Initiative, Bangkok, Thailand
