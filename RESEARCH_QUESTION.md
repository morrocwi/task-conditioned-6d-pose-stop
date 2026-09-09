# Research question and falsifier

## Core question

Can an iterative 6D pose estimator terminate refinement because the current pose evidence is already sufficient for the downstream manipulation task, even when the estimator itself has not reached its generic convergence criterion?

Let `k_T` be the earliest task-sufficient refinement stage and `k_E` the estimator-side stopping stage.

The computational opportunity exists when:

```text
k_T < k_E
```

## Real-backend hypotheses

H1. Task-conditioned stopping reduces mean refinement stages relative to estimator-side stopping.

H2. Task-conditioned stopping reduces end-to-end perception latency after including uncertainty and gate overhead.

H3. Task completion is non-inferior within a predeclared margin.

## Falsifiers

The proposal is not supported for a tested backend/task if any of the following hold:

- `k_T < k_E` is rare or absent;
- total latency is not reduced after gate/uncertainty overhead;
- task completion drops beyond the declared margin;
- HOLD frequency makes overall completion operationally unacceptable;
- the stopping signal is not sufficiently calibrated to separate ready from not-ready episodes.
