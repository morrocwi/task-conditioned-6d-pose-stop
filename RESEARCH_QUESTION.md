# Research question and falsifier

## Core question

Can an iterative 6D pose estimator terminate refinement because the current pose evidence is already sufficient for the downstream manipulation task, even when the estimator itself has not reached its generic convergence criterion?

Let `k_T` be the earliest task-sufficient refinement stage and `k_E` the estimator-side stopping stage.

```text
k_T < k_E
```

## Evidence status

### Supported in the public numerical fixture

- Iterative rigid registration is executed rather than represented only by synthetic action costs.
- The gate is task-conditioned and cannot access hidden ground-truth pose error.
- On the frozen in-distribution standard run, mean `k_T` is below mean `k_E` for all three tasks, with paired bootstrap intervals below zero.
- Completion, ACT, HOLD, unsafe ACT, iterations, and wall-clock readouts are all reported.
- A 2x observation-noise stress can substantially reduce the usefulness of the frozen stopping rule.

### Still open for a real vision/robotics backend

H1. Task-conditioned stopping reduces refinement stages relative to estimator-side stopping.

H2. Task-conditioned stopping reduces end-to-end perception latency after uncertainty/gate overhead.

H3. Task completion is non-inferior within a predeclared margin.

## Falsifiers

For a tested real backend/task, the proposed practical advantage is unsupported if:

- `k_T < k_E` is rare or absent;
- total latency is not reduced after gate/uncertainty overhead;
- task completion drops beyond the declared margin;
- HOLD frequency makes overall completion operationally unacceptable;
- unsafe ACT exceeds the declared operating tolerance;
- the stopping signal fails under realistic calibration or distribution shift.

A numerical result cannot falsify or confirm the real-camera/GPU/robot hypotheses by itself.
