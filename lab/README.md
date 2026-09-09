# University Lab Protocol: Real-System Test

This folder is the backend-agnostic entry point for testing **coverage-qualified downstream task stopping** on a real iterative 6D pose system.

The single research question is:

\[
k_C < k_E\;?
\]

- `k_C`: first refinement stage whose **calibrated completion envelope is entirely PASS** for the declared downstream task.
- `k_E`: estimator-side stopping stage.

A lab does **not** need to use FoundationPose. Any iterative 6D backend can be tested if it can expose a refinement trajectory.

## What a lab needs

1. A real iterative 6D pose backend.
2. Independent pose ground truth for TRAIN/CALIBRATION/TEST trials (motion capture, calibrated fiducials, synthetic ground truth, or another predeclared independent reference).
3. A declared task-admissibility reader in 6D pose-error coordinates.
4. Measured per-stage latency on the actual hardware.

Translation units are metres. Rotation units are radians.

## Minimum workflow

### 1. Freeze the experiment before final test

Predeclare:

- backend name/version/commit and model weights;
- hardware;
- object/task set;
- observable features exported at each refinement stage;
- nominal coverage `1-alpha`;
- non-inferiority margin;
- TRAIN, CALIBRATION, and FINAL TEST split rule;
- estimator-side stopping rule;
- task reader.

Do not modify these after opening FINAL TEST without starting a new final-test lineage.

### 2. Run the full refinement trajectory once per perception trial

For each episode, save every stage from the same run. Truncating a full trajectory at stage `k` is a valid counterfactual for perception latency only when later refinement does not change earlier stages.

Export JSONL with this shape:

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

`features` must contain **only quantities available online before deciding whether to continue**. Ground truth belongs only in `oracle`.

### 3. Keep splits disjoint

Use three separate files:

```text
train.jsonl
calibration.jsonl
test.jsonl
```

The harness rejects duplicate `episode_id` values across splits.

TRAIN fits the observable error-shape model.

CALIBRATION computes one whole-trajectory conformal score per episode.

FINAL TEST is evaluated only after the model and calibration scale are frozen.

### 4. Run the common evaluator

```bash
python lab/run_real_system.py \
  --config lab/config.example.json \
  --train path/to/train.jsonl \
  --calibration path/to/calibration.jsonl \
  --test path/to/test.jsonl \
  --out lab_results.json
```

The evaluator reports:

- whole-trajectory envelope coverage;
- `k_C < k_E` rate;
- paired `k_C-k_E` bootstrap interval;
- task completion for certificate-stop and estimator-stop;
- HOLD rate;
- unsafe ACT rate;
- unsafe ACT on covered episodes;
- measured perception latency difference;
- paired completion difference;
- non-inferiority decision against the predeclared margin.

## What counts as a successful real-backend test

A paper should not define success using one number. Report the complete tuple:

```text
coverage
k_C < k_E
paired stage difference
paired latency difference
completion difference
HOLD
unsafe ACT
```

A useful result requires, at minimum:

1. the calibrated set has empirically reported held-out coverage;
2. `k_C` is earlier than `k_E` often enough to matter;
3. measured perception latency is lower after gate overhead;
4. completion is not worse than the predeclared margin;
5. HOLD is operationally acceptable;
6. unsafe ACT is reported, never hidden inside success-given-ACT.

## Perception validation vs physical-robot validation

This harness can establish a **real-backend perception result** when pose ground truth is measured on real sensor data.

It does **not** by itself establish physical manipulation non-inferiority.

For a physical-robot claim, run a separate execution study in which the actual robot executes the compared stopping policies under a predeclared randomized or blocked design. Record true physical task completion as an independent outcome.

## Built-in task readers

The public evaluator currently supports:

- `box`: every selected pose-error coordinate must lie within its declared tolerance;
- `l1`: the normalized sum of selected absolute pose errors must be at most one.

For tasks that cannot be represented this way, implement a custom reader rather than forcing the task into an invalid box.

## Smoke test

Anyone can verify the harness without hardware:

```bash
python lab/make_example_data.py --out-dir artifacts/lab-example --n 40
python lab/run_real_system.py \
  --config lab/config.example.json \
  --train artifacts/lab-example/train.jsonl \
  --calibration artifacts/lab-example/calibration.jsonl \
  --test artifacts/lab-example/test.jsonl \
  --out artifacts/lab-example/results.json
```

The example data are only a software smoke test. They are not research evidence.

## Evidence boundary

Passing this harness does not certify a robot, a pose estimator, or a safety claim. It standardizes the test so different laboratories can expose exactly where the proposal works, fails, or becomes too conservative.
