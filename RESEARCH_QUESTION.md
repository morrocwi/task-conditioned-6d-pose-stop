# Research question and falsifier

## Core question

Can iterative 6D pose refinement terminate before estimator-side convergence because a **calibrated completion envelope** is already invariant under the downstream task reader?

Let `C_hat_k^cal` be the retained calibrated completion envelope at refinement stage `k`, and let `O_T` be the downstream task reader.

Define the earliest coverage-qualified task-sufficient stage

```text
k_C = min{k : O_T(z) = PASS for every z in C_hat_k^cal}
```

and let `k_E` be estimator-side stopping.

The computational opportunity is:

```text
k_C < k_E
```

This is stronger than stopping on a raw uncertainty score. The still-admissible pose alternatives must be task-equivalent under a completion envelope whose coverage is independently calibrated.

## Numerical hypotheses

N1. Under the declared in-distribution numerical generator, the frozen trajectory-calibrated envelope achieves held-out episode-level coverage near its predeclared target.

N2. `k_C < k_E` occurs on a substantial fraction of held-out episodes for at least one declared task.

N3. Coverage-qualified task stopping reduces mean refinement stage relative to estimator-side stopping without an unacceptable loss of overall task completion.

N4. Under a declared distribution shift, envelope coverage and/or operational completion may degrade; the method must report that degradation rather than silently transfer the in-distribution guarantee.

## Real-backend hypotheses

H1. A real iterative 6D pose backend can expose an observable proxy from which a calibrated completion envelope can be constructed without giving hidden ground truth to the online gate.

H2. Coverage-qualified task stopping reduces mean refinement stages relative to estimator-side stopping.

H3. Coverage-qualified task stopping reduces end-to-end perception latency after calibration-adapter and gate overhead.

H4. Overall task completion is non-inferior within a predeclared margin, with HOLD retained in the denominator.

## Falsifiers

The proposal is not supported for a tested backend/task if any of the following hold:

- held-out in-distribution completion-envelope coverage materially misses the predeclared target;
- `k_C < k_E` is rare or absent;
- total latency is not reduced after uncertainty/calibration/gate overhead;
- task completion drops beyond the declared margin;
- HOLD frequency makes overall completion operationally unacceptable;
- distribution shift causes coverage failure that is not detected or reported;
- the observable proxy cannot support a useful calibrated envelope;
- the downstream task reader is misspecified so that reader-invariance does not track real task success.

A numerical result cannot confirm the real-camera/GPU/robot hypotheses by itself.
