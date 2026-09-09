# Coverage-Qualified Task Stopping for Iterative 6D Pose Refinement

Public, implementation-first research repository.

> **Question:** can iterative pose refinement stop before estimator-side convergence because every pose still admitted by a calibrated completion envelope already gives the same downstream task verdict?

The central event is:

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

The repository contains five explicit layers:

1. **Finite diagnostic** — exhaustive toy routing fixture (`benchmark.py`).
2. **Numerical 6D backend** — iterative point-to-point ICP/Kabsch on generated 3-D point clouds (`experiments/numerical_icp.py`).
3. **Raw-proxy completion ablation** — trajectory-level split calibration directly on the local proxy (`experiments/conformal_completion.py`).
4. **Learned-shape calibrated certificate** — TRAIN a fixed observable error-shape model, CALIBRATE one conformal score per whole refinement episode, then evaluate on new FINAL TEST seeds (`experiments/learned_conformal_completion.py`).
5. **Repeated-cycle replication** — rerun the frozen TRAIN→CALIBRATION→TEST procedure across independent numerical draws (`experiments/replication_study.py`).

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

## First frozen learned-shape final test

The first FINAL TEST was opened only after the learned-shape method and replacement final-test seeds were committed. Earlier development test seeds were retired and are recorded in the result lineage.

Protocol:

```text
TRAIN        160 episodes
CALIBRATION  160 episodes
FINAL TEST   120 episodes
STRESS        60 episodes (2x sensor noise)
nominal marginal whole-trajectory coverage = 90%
```

Observed whole-trajectory envelope coverage was **85.00%** (Wilson 95% CI 77.53%–90.30%). This is below the nominal 90% target and is reported as such.

| Task | estimator mean k | completion mean k | `k_C < k_E` | estimator completion | completion-stop completion | HOLD | unsafe ACT |
|---|---:|---:|---:|---:|---:|---:|---:|
| top suction | 13.500 | 10.350 | 90.00% | 95.00% | 92.50% | 7.50% | 0.00% |
| label alignment | 13.500 | 10.900 | 87.50% | 92.50% | 91.67% | 8.33% | 0.00% |
| keyed insertion | 13.500 | 11.750 | 87.50% | 90.00% | 90.00% | 10.00% | 0.00% |

Frozen record: [`results/learned_conformal/STANDARD_RESULTS.md`](results/learned_conformal/STANDARD_RESULTS.md).

On this runner, numerical latency also fell with the earlier stopping stage, but these are machine-specific CPU numerical timings — **not GPU/robot speedups**.

## Repeated complete-cycle evidence

Because one realized test proportion is not the conformal theorem, the frozen learned procedure was rerun through **10 independent TRAIN→CALIBRATION→TEST cycles** without retuning the method.

Across those cycles:

```text
nominal coverage                 90.00%
mean realized trajectory cover  91.17%
median realized coverage        91.67%
range                            83.33%–95.00%
replicates >= nominal           9 / 10
```

Task utility averaged:

| Task | mean `k_C < k_E` | mean `k_C-k_E` | mean completion difference | mean HOLD | mean unsafe ACT |
|---|---:|---:|---:|---:|---:|
| top suction | 92.33% | -3.228 | -1.17 pp | 6.00% | 0.00% |
| label alignment | 91.33% | -2.770 | 0.00 pp | 6.17% | 0.00% |
| keyed insertion | 83.83% | -1.632 | -0.33 pp | 6.50% | 0.00% |

Frozen replication record: [`results/replication/STANDARD_REPLICATION.md`](results/replication/STANDARD_REPLICATION.md).

This repeated numerical study makes the interpretation more precise: the 85% first final-test realization is not evidence by itself that the procedure's long-run marginal coverage is 85%. Across these 10 independent numerical cycles the mean realized proportion was 91.17%, while individual cycles still varied materially. Ten cycles do not establish real-sensor coverage or a universal guarantee.

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

For the larger numerical experiment:

```bash
python reproduce.py --profile standard
```

Outputs include:

```text
artifacts/numerical/
artifacts/conformal-raw/
artifacts/conformal-learned/
artifacts/replication/
```

## Automatic checks

GitHub Actions rebuilds the experiment on Linux, macOS, and Windows. A separate Ubuntu standard-evidence job runs the larger learned certificate plus the 10-cycle replication study and uploads the result records.

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
- repeated complete-cycle calibration/test diagnostics without method tuning;
- glosa claim-card validation.

The current public CI has passed on Ubuntu, macOS, and Windows for the full numerical stack.

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

Current evidence ceiling: **K1 public provisional / I4 mechanical-original-record checks**. Repeated numerical cycles increase mechanical stress-testing but do not constitute independent external-human or physical-robot replication.

## Claim boundary

Supported now:

- a public iterative 6D numerical backend can be stopped by task-reader invariance over a calibrated completion set;
- the first frozen learned final test shows `k_C < k_E` in 87.5–90% of episodes depending on task;
- across 10 complete-cycle numerical replications, mean realized trajectory coverage is 91.17% with 9/10 cycles at or above the nominal 90% proportion;
- across those cycles, completion-stop mean task completion differs from estimator stopping by 0 to -1.17 percentage points depending on task;
- the implemented robust gate produced zero unsafe ACTs in the reported numerical studies;
- a naive high-coverage set can be operationally useless because it is too wide;
- shifted conditions can force task-specific HOLD.

Still open / HOLD:

- real RGB-D coverage;
- FoundationPose-specific or GPU speedup;
- physical manipulation non-inferiority;
- physical safety certification;
- validity under arbitrary distribution shift;
- IDM-specific advantage.

## Prior-art boundary

The repository does **not** claim to invent task-aware perception, early stopping, conformal prediction, or conformal 6D pose uncertainty. Relevant prior work includes SPADE for task-relevant perception cost, conformal pose uncertainty by Yang & Pavone (CVPR 2023), deterministic conformal 6D confidence regions by Wang et al. (ICCV 2025), and estimator-side early stopping in AnyBox.

The narrow candidate contribution is:

> **coverage-qualified downstream task stopping of iterative 6D pose refinement: stop when the calibrated pose completion set has collapsed to one task-reader verdict, not merely when the estimator has converged.**

## Author

Yaoharee Lahtee  
Open Civil Science Initiative, Bangkok, Thailand
