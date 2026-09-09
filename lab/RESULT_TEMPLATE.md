# External Lab Result Report

Use this template unchanged where possible so results from different laboratories remain comparable.

## Laboratory and system

- Institution:
- Laboratory:
- Date:
- Backend name:
- Backend version / git commit:
- Model weights identifier:
- Camera / sensor:
- GPU / CPU:
- Operating system:
- Object set:
- Task set:
- Ground-truth source:
- SE(3) error convention:

## Frozen protocol

- TRAIN episodes:
- CALIBRATION episodes:
- FINAL TEST episodes:
- Split rule:
- `alpha`:
- nominal whole-trajectory coverage:
- non-inferiority margin:
- estimator-side stopping rule:
- online feature vector definition:
- task reader definition:
- deviations from `lab/README.md`:

## Primary results

| Task | test coverage | `k_C<k_E` | mean `k_C-k_E` [95% CI] | estimator completion | certificate completion | completion difference [95% CI] | HOLD | unsafe ACT | perception latency difference [95% CI] | non-inferiority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| task 1 | | | | | | | | | | |

## Mandatory failure accounting

- Number of unsafe ACTs on covered episodes:
- Number of unsafe ACTs total:
- Number of HOLD episodes:
- Cases where `k_C>=k_E`:
- Calibration/test coverage shortfall, if any:
- Distribution-shift conditions tested:

## Claim form

A permitted real-backend claim should be bounded to the measured system, for example:

> On [backend/version/hardware] under [task/data condition], coverage-qualified stopping occurred before estimator-side stopping in [X%] of held-out trials. The paired perception-latency difference was [D, 95% CI], task completion differed by [E, 95% CI] under a predeclared non-inferiority margin of [M], with HOLD [H%] and unsafe ACT [U%].

Do not replace this with a universal claim about 6D pose estimation.

## Physical robot extension

If physical manipulation was executed, additionally report:

- robot and controller;
- randomization/blocking scheme;
- number of physical executions per policy;
- independent physical success criterion;
- policy-level physical completion and confidence intervals;
- intervention/emergency-stop count;
- excluded trials with reasons.

Pose-ground-truth replay alone is not a physical-robot result.
