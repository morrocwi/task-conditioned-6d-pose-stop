# Real-Backend Protocol v1.0 — University Lab Test

The implementation entry point is [`../lab/README.md`](../lab/README.md).

## Single testable claim

For an iterative 6D pose backend, define:

\[
k_C=\min\{k:\mathcal C_k^{cal}\subseteq\mathcal A_T\}
\]

where `C_k^cal` is the calibrated pose-error completion envelope at refinement stage `k`, and `A_T` is the declared downstream task-admissible set.

Let `k_E` be the backend's own estimator-side stopping stage.

The real-system question is:

\[
\boxed{k_C<k_E?}
\]

and, if so, whether the earlier stop reduces measured perception latency without reducing task completion beyond a predeclared margin.

## Required experimental roles

Use disjoint:

```text
TRAIN
CALIBRATION
FINAL TEST
```

TRAIN fits the mapping from online-observable refinement features to pose-error shape.

CALIBRATION freezes the whole-trajectory conformal scale.

FINAL TEST is opened only after the backend, features, task reader, alpha, estimator stopping rule, and non-inferiority margin are frozen.

## Required backend instrumentation

At every refinement stage export:

- stage index `k`;
- a numeric feature vector containing only online-observable quantities;
- measured incremental wall-clock time;
- whether the backend's native stopping rule fires.

Independently record the 6D pose error relative to a predeclared ground-truth source. Translation is reported in metres and rotation in radians.

The public schema is [`../lab/episode.schema.json`](../lab/episode.schema.json), and an integration skeleton is [`../lab/ADAPTER_TEMPLATE.py`](../lab/ADAPTER_TEMPLATE.py).

## Required baselines

The minimum comparison is:

1. estimator-side stopping;
2. coverage-qualified completion stopping.

A paper may additionally report fixed-iteration and full-budget baselines, but they do not replace the direct comparison above.

All policies must use the same initial observation and same refinement trajectory prefix.

## Required output

Report all of:

- held-out whole-trajectory envelope coverage;
- `k_C < k_E` rate;
- paired confidence interval for `k_C-k_E`;
- certificate-stop and estimator-stop task completion;
- predeclared non-inferiority margin and decision;
- HOLD rate;
- unsafe ACT rate;
- unsafe ACT on covered episodes;
- measured perception latency for both policies;
- paired latency difference including gate overhead.

Do not report success-given-ACT without also reporting HOLD and overall completion.

## Minimum claim discipline

### Real-backend perception claim

Permitted when the test uses real sensor/backend trajectories, independently measured pose ground truth, and actual hardware timing.

Example claim form:

> On backend X under protocol Y, coverage-qualified stopping occurred before estimator-side stopping in Z% of held-out trials and reduced measured perception latency by D while task-admissibility completion differed by E under the declared margin.

### Physical manipulation claim

Not licensed by pose-ground-truth replay alone.

A physical manipulation non-inferiority claim requires actual robot executions under a predeclared randomized or blocked comparison. The physical task outcome must be measured independently of the stopping gate.

## Distribution shift

The split-conformal interpretation applies only under the declared exchangeability conditions. Any new camera, object family, lighting regime, sensor-noise regime, or backend version that materially changes the data-generating process must be reported as a new condition and may require recalibration.

## Falsifiers

The practical proposal is unsupported for a tested backend/task if any of the following holds:

- `k_C < k_E` is rare or absent;
- measured end-to-end perception latency does not decrease after gate overhead;
- completion falls beyond the predeclared margin;
- HOLD makes the method operationally unusable;
- unsafe ACT is unacceptably high;
- held-out coverage is inadequate for the intended operating point;
- the calibrated envelope collapses under realistic distribution shift.

## One-command evaluator

```bash
python lab/run_real_system.py \
  --config lab/config.example.json \
  --train train.jsonl \
  --calibration calibration.jsonl \
  --test test.jsonl \
  --out lab_results.json
```

This command is backend-agnostic. A university laboratory can test FoundationPose, ICP, a learned pose refiner, or another iterative 6D system by adapting only the data-export layer.
