# Coverage-Qualified Task Stopping for Iterative 6D Pose Refinement

Public, implementation-first research repository.

> **Question:** can iterative pose refinement stop before estimator-side convergence because every pose still admitted by a calibrated completion envelope already gives the same downstream task verdict?

The central event is now:

```text
k_C < k_E
```

where `k_C` is the first stage whose **calibrated completion envelope is entirely PASS under the declared task reader**, and `k_E` is estimator-side convergence.

This is intentionally stronger than stopping on a raw confidence/uncertainty threshold.

## Core distinction

```text
pose not known exactly
        !=
task verdict unresolved
```

A robot does not need latent pose identity if the unresolved alternatives no longer contain a distinction that can change the declared task outcome.

The online rule is:

```text
observable refinement readout
        -> calibrated completion envelope
        -> downstream task reader
        -> ACT | CONTINUE | HOLD
```

ACT is licensed only when **every pose error in the retained completion set passes the task reader**.

## Evidence layers

The repository now contains four explicit layers:

1. **Finite diagnostic** — exhaustive toy routing fixture (`benchmark.py`).
2. **Numerical 6D backend** — iterative point-to-point ICP/Kabsch on generated 3-D point clouds (`experiments/numerical_icp.py`).
3. **Raw-proxy completion ablation** — trajectory-level split calibration directly on the local proxy (`experiments/conformal_completion.py`).
4. **Learned-shape calibrated certificate** — TRAIN a fixed observable error-shape model, CALIBRATE one conformal score per whole refinement episode, then evaluate on new FINAL TEST seeds (`experiments/learned_conformal_completion.py`).

No camera, RGB-D sensor, neural pose model, contact physics, or physical robot is executed by these numerical experiments.

## Why the completion envelope exists

The Toledo-derived project construction is documented in [`theory/TOLEDO_COMPLETION_ENVELOPE.md`](theory/TOLEDO_COMPLETION_ENVELOPE.md). The robotics-specific equations are explicitly labeled as **NEW project definitions/derivations**, not existing Toledo mathematics.

The key set is

```text
C_k = latent pose completions not yet excluded by the retained evidence
```

and the task verdict is robust only when the task reader is invariant over that set.

A statistical bridge is then used to obtain a calibrated numerical envelope from observable refinement features. See [`theory/COVERAGE_TO_TASK_CERTIFICATE.md`](theory/COVERAGE_TO_TASK_CERTIFICATE.md).

Conformal pose uncertainty itself is prior art. The research question here is the **use of a calibrated pose set as a downstream task stopping certificate for iterative refinement**.

## Frozen learned-shape result

The first frozen FINAL TEST was opened only after the learned-shape method and replacement final-test seeds were committed. Earlier development test seeds were retired and are recorded in the result lineage.

Protocol:

```text
TRAIN        160 episodes
CALIBRATION  160 episodes
FINAL TEST   120 episodes
STRESS        60 episodes (2x sensor noise)
nominal marginal whole-trajectory coverage = 90%
```

### In-distribution final test

Observed whole-trajectory envelope coverage: **85.00%** (Wilson 95% CI 77.53%–90.30%).

This is below the nominal 90% target and is reported as such. It must not be rewritten as “90% empirical coverage”.

| Task | estimator mean k | completion mean k | `k_C < k_E` | estimator completion | completion-stop completion | HOLD | unsafe ACT |
|---|---:|---:|---:|---:|---:|---:|---:|
| top suction | 13.500 | 10.350 | 90.00% | 95.00% | 92.50% | 7.50% | 0.00% |
| label alignment | 13.500 | 10.900 | 87.50% | 92.50% | 91.67% | 8.33% | 0.00% |
| keyed insertion | 13.500 | 11.750 | 87.50% | 90.00% | 90.00% | 10.00% | 0.00% |

Frozen record: [`results/learned_conformal/STANDARD_RESULTS.md`](results/learned_conformal/STANDARD_RESULTS.md).

On this runner, numerical latency also fell with the earlier stopping stage (16.60 ms estimator-stop versus 13.26 / 13.89 / 14.89 ms completion-stop), but these are machine-specific CPU numerical timings — **not GPU/robot speedups**.

## A useful negative result

The simpler raw-proxy conformal construction achieved **95.83%** held-out whole-trajectory coverage in its frozen run but returned **HOLD for 100%** of episodes for all three tasks.

That ablation establishes an important design constraint:

```text
coverage alone != useful stopping certificate
```

The set must be both:

```text
sufficiently covered
AND
sufficiently task-discriminative
```

The learned-shape model makes the set substantially more informative, but its first frozen final test observed only 85% trajectory coverage against a nominal 90% target. The coverage–utility problem therefore remains an empirical part of the paper rather than being hidden as an implementation detail.

## Distribution-shift boundary

Under 2x sensor noise, the learned envelope remained usable for top suction and label alignment in this numerical fixture, but keyed insertion returned HOLD for every stress episode.

That failure is retained deliberately. A guarantee derived under exchangeability is not silently transferred to shifted conditions.

## Run it yourself

Requires Python 3.10+.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python reproduce.py --profile quick
```

For the frozen-size numerical experiment:

```bash
python reproduce.py --profile standard
```

Outputs include:

```text
artifacts/numerical/
artifacts/conformal-raw/
artifacts/conformal-learned/
```

## Automatic checks

GitHub Actions rebuilds the experiment on Linux, macOS, and Windows. A separate Ubuntu standard-evidence job runs the larger learned certificate and uploads the result record.

Mechanical controls include:

- exact Kabsch recovery and pose identity;
- task-switch and coupled-task counterexamples;
- no hidden-ground-truth object in the online gate signature;
- disjoint TRAIN / CALIBRATION / FINAL TEST seeds;
- retirement of development test seeds after they were inspected;
- split-conformal order-statistic checks;
- whole-trajectory rather than selected-stage calibration;
- robust-PASS set-inclusion tests;
- `unsafe_act_on_covered_episode_count == 0` for the implemented numerical readers;
- glosa claim-card validation.

## Evidence discipline

The repository applies the five-question / E-A-D discipline from **glosa — Rigour Without Infrastructure**:

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

- a public iterative 6D numerical backend can be stopped by task-reader invariance over a calibrated completion set;
- on the first frozen learned-shape final test, `k_C < k_E` occurs in 87.5–90% of episodes depending on task;
- the implemented robust gate produced no unsafe ACTs in that numerical final test;
- a naive high-coverage set can be operationally useless because it is too wide;
- shifted conditions can force task-specific HOLD.

Still open / HOLD:

- empirical attainment of the nominal coverage target across repeated independent calibration/test draws;
- real RGB-D coverage;
- FoundationPose-specific or GPU speedup;
- physical manipulation non-inferiority;
- physical safety certification;
- IDM-specific advantage.

## Prior-art boundary

The repository does **not** claim to invent task-aware perception, early stopping, conformal prediction, or conformal 6D pose uncertainty. Relevant prior work includes SPADE for task-relevant perception cost, conformal pose uncertainty by Yang & Pavone (CVPR 2023), deterministic conformal 6D confidence regions by Wang et al. (ICCV 2025), and estimator-side early stopping in AnyBox.

The narrow candidate contribution is:

> **coverage-qualified downstream task stopping of iterative 6D pose refinement: stop when the calibrated pose completion set has collapsed to one task-reader verdict, not merely when the estimator has converged.**

## Author

Yaoharee Lahtee  
Open Civil Science Initiative, Bangkok, Thailand
