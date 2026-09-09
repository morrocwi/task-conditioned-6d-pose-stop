# University Lab Protocol — Real Iterative 6D Pose Test

This folder is the backend-agnostic entry point for testing **coverage-qualified downstream task stopping** on a real iterative 6D pose system.

The primary event is

\[
k_C < k_E,
\]

where:

- `k_C` is the first stage whose calibrated completion envelope is entirely PASS for the declared task reader;
- `k_E` is estimator-side stopping.

If no certificate is reached within the refinement budget, `k_C` is **undefined / +infinity**. The executed policy may continue to the terminal budget and return HOLD; that terminal endpoint must not be reported as `k_C`.

A lab does not need FoundationPose. Any iterative 6D backend can be tested if it exposes a refinement trajectory.

## What a lab needs

1. A real iterative 6D pose backend.
2. Independent pose ground truth for TRAIN/CALIBRATION/FINAL TEST trials: motion capture, calibrated fiducials, simulation ground truth, or another predeclared reference.
3. A declared task-admissibility reader in 6D pose-error coordinates.
4. Online-observable features available before the continue/stop decision.
5. A predeclared statistical plan.
6. For a speed claim, direct paired timing of the actual stopped policies on the target hardware.

Translation units are metres. Rotation units are radians. The reference helper reports components of the **relative rotation vector**; these are not globally valid Euler roll/pitch/yaw angles.

## 1. Freeze the experiment before FINAL TEST

Predeclare:

- backend name/version/commit and model weights;
- hardware, software stack, synchronization method, warm-up policy;
- object/task set and independent sampling unit;
- observable feature vector;
- nominal coverage `1-alpha`;
- task reader and tolerances;
- estimator-side stopping rule;
- non-inferiority margin;
- confidence level;
- minimum FINAL TEST sample size / power rationale;
- TRAIN, CALIBRATION, FINAL TEST split rule;
- timing mode.

Do not modify these after opening FINAL TEST without starting a new final-test lineage.

## 2. Export a full refinement trajectory

For model fitting and counterfactual prefix analysis, export every stage from the same run:

```json
{
  "episode_id": "test-0001",
  "stages": [
    {
      "k": 0,
      "features": [0.0, -5.2, 0.71],
      "incremental_ms": 12.8,
      "estimator_stop": false
    }
  ],
  "oracle": {
    "abs_pose_error_6d": [
      [0.003, 0.002, 0.004, 0.03, 0.02, 0.08]
    ]
  }
}
```

`features` must contain only quantities available online before deciding whether to continue. Ground truth belongs only in `oracle`.

Truncating a fully generated trajectory at stage `k` provides a **trajectory-prefix cost estimate**. It is not, by itself, a measured speedup of an independently stopped online policy.

## 3. Keep TRAIN / CALIBRATION / FINAL TEST disjoint

Use:

```text
train.jsonl
calibration.jsonl
test.jsonl
```

The evaluator rejects duplicate episode IDs across splits.

- TRAIN fits the observable error-shape model.
- CALIBRATION computes one whole-trajectory conformal score per independent episode.
- FINAL TEST is evaluated only after model, calibration rule, task reader, and statistical plan are frozen.

For finite split conformal calibration, the public implementation uses the augmented order statistic. If the required rank is `n+1`, the quantile is `+infinity`, producing an uninformative envelope and fail-closed HOLD. It never silently substitutes the largest finite score.

## 4. Fail-closed certificate behavior

The public evaluator treats corrupt certificate numerics conservatively:

```text
NaN / invalid model / malformed tolerance / overflow
        -> unbounded certificate
        -> no ACT
        -> CONTINUE or HOLD
```

Invalid numerics must never collapse uncertainty to zero.

## 5. Run the common evaluator

```bash
python lab/run_real_system.py \
  --config lab/config.example.json \
  --train path/to/train.jsonl \
  --calibration path/to/calibration.jsonl \
  --test path/to/test.jsonl \
  --out lab_results.json
```

The default config uses:

```text
timing_mode = trajectory_prefix_estimate
```

In this mode the evaluator reports paired prefix-cost differences but sets `latency_reduction_pass = null`. A speed claim is deliberately disabled.

## 6. Statistical decision

The evaluator reports the paired completion difference and a conservative finite-sample non-inferiority lower confidence bound based on paired discordances (`candidate-only success` versus `comparator-only success`) with Clopper–Pearson bounds and Bonferroni allocation.

The decision requires the predeclared:

```text
margin
confidence level
minimum test episodes
independent sampling unit
```

A single pair of equal outcomes cannot certify population non-inferiority. If the declared minimum sample size is not reached, the result remains insufficient.

## 7. Real speed claims require online-policy timing

To claim latency reduction, use:

```json
"timing_mode": "online_policy_measured"
```

and export paired policy timing for each task/episode:

```json
"policy_timing_ms": {
  "top_suction": {
    "certificate_stop": 21.4,
    "estimator_stop": 33.9
  }
}
```

Those measurements must come from actual separately executed policy loops and include all policy-specific work required for the decision, including feature extraction, certificate/gate computation, refinement updates, synchronization, and relevant device timing. Use warm-up and a predeclared paired timing design.

Only `online_policy_measured` mode may emit a latency-reduction decision. Prefix replay remains descriptive.

## 8. Required output

Report the complete tuple:

```text
held-out trajectory coverage
certificate rate
k_C < k_E rate (all assigned episodes)
k_C distribution conditional on certificate
executed endpoint distribution
paired prefix cost and/or measured policy time
completion difference
non-inferiority lower bound and decision
HOLD
unsafe ACT
unsafe ACT on covered episodes
```

HOLD remains in the overall completion denominator.

## 9. Real-backend vs physical-robot claims

A real-sensor/backend experiment with independent pose ground truth can support a **real-backend perception result**.

It does not establish physical manipulation non-inferiority. For that claim, execute the compared policies on the actual robot under a predeclared randomized or blocked design and measure physical task completion independently of the stopping gate.

## 10. Built-in readers

The evaluator currently supports:

- `box`: every selected relative-pose error component must lie within tolerance;
- `l1`: normalized selected absolute relative-pose errors must sum to at most one.

If a task cannot be represented faithfully by these readers, implement and validate a custom reader rather than forcing it into an invalid box.

## Smoke test

```bash
python lab/make_example_data.py --out-dir artifacts/lab-example --n 40
python lab/run_real_system.py \
  --config lab/config.example.json \
  --train artifacts/lab-example/train.jsonl \
  --calibration artifacts/lab-example/calibration.jsonl \
  --test artifacts/lab-example/test.jsonl \
  --out artifacts/lab-example/results.json
```

The generated example is only a software smoke test, not research evidence.

## Evidence boundary

Passing the harness does not certify a robot, estimator, task reader, or safety property. It standardizes a falsifiable cross-lab test and forces insufficient calibration, invalid numerics, inadequate sample size, and missing online timing to remain explicit rather than being silently promoted into stronger claims.
