# Real-backend protocol v0.1

## Goal

Test whether downstream task sufficiency occurs before estimator-side convergence on one real iterative 6D pose backend.

## Minimal system

Action set:

```text
ACT
CONTINUE_REFINEMENT
HOLD
```

Do not begin with coordinate-selective refinement unless the backend exposes a real operator whose compute savings can be measured.

## Required baselines

1. Fixed refinement count.
2. Full configured refinement budget.
3. Estimator-side early stopping.
4. Task-conditioned early stopping.

All conditions must start from the same initial observation and initial pose inference.

## Required independent measurements

The stopping gate must not define the test outcome. Evaluate downstream success independently using ground-truth pose, fiducial measurement, task completion, or another predeclared criterion.

## Calibration split

Separate data into at least:

- calibration/development split for uncertainty and task gate;
- locked test split for final evaluation.

Do not fit stopping thresholds on final test outcomes.

## Timing

Record:

- initial inference time;
- uncertainty computation time;
- gate time;
- each refinement iteration time;
- total perception time;
- downstream execution time if relevant.

Only total pipeline time licenses a speed claim.

## Outcomes

Report:

- overall task completion over all assigned episodes;
- success conditional on ACT;
- HOLD rate;
- unsafe ACT rate;
- mean/median/p95 refinement stages;
- mean/median/p95 total perception latency;
- fraction of episodes with `k_T < k_E`, `k_T = k_E`, and apparent early stop followed by failure.

## Statistical decision

Predeclare the non-inferiority margin and confidence procedure before final test evaluation. Point-estimate similarity is not evidence of non-inferiority.
