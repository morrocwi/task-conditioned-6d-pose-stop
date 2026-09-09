# Frozen standard result — learned-shape trajectory-conformal completion certificate

**Evidence label:** `[NumericalBackend+LearnedShape+SplitConformal]`  
**First frozen final-test run:** GitHub Actions run `34302619781`  
**Evaluated commit:** `0eb47e0537535b92b49967f89b130fe1fdd2a531`  
**Python:** 3.11.16 on Ubuntu 24.04 runner  
**NumPy:** 2.4.6

This file records the first public execution after the learned-shape method and new final-test seeds were frozen. The previous development test seeds `2026090931..33` were explicitly retired and were not reused as final-test seeds.

## Protocol

- TRAIN episodes: 160 (`TRAIN_SEED=2026090920`)
- CALIBRATION episodes: 160 (`CAL_SEED=2026090951`)
- target marginal whole-trajectory coverage: 90% (`alpha=0.10`)
- FINAL TEST: 120 episodes across seeds `2026090961, 2026090962, 2026090963`
- shifted STRESS: 60 episodes (`STRESS_SEED=2026090993`, 2x sensor noise)
- maximum refinement stage: 20
- points per generated cloud: 120

The conformal episode score is the maximum log-residual across all retained stages and all six pose-error coordinates. The online gate does not receive hidden ground-truth pose.

## Calibration

- conformal rank: 145 / 160
- frozen `q`: 1.9851453152702532
- calibration median score: 1.5653911041058104
- calibration max score: 3.0005182862837256

## In-distribution final test

Whole-trajectory envelope coverage was **85.00%** (102/120), Wilson 95% CI **77.53%–90.30%**.

This observed proportion is below the nominal 90% target. The interval includes 90%, but this run must not be described as empirically achieving 90% coverage.

| Task | estimator mean k | completion mean k | `k_C < k_E` | estimator completion | completion-stop completion | HOLD | unsafe ACT |
|---|---:|---:|---:|---:|---:|---:|---:|
| top suction | 13.500 | 10.350 | 90.00% | 95.00% | 92.50% | 7.50% | 0.00% |
| label alignment | 13.500 | 10.900 | 87.50% | 92.50% | 91.67% | 8.33% | 0.00% |
| keyed insertion | 13.500 | 11.750 | 87.50% | 90.00% | 90.00% | 10.00% | 0.00% |

Mean measured numerical latency on this runner:

| Task | estimator-stop | completion-stop |
|---|---:|---:|
| top suction | 16.597 ms | 13.258 ms |
| label alignment | 16.597 ms | 13.892 ms |
| keyed insertion | 16.597 ms | 14.888 ms |

Timing is machine-specific and is not a GPU or robot speed result.

For all three task readers, `unsafe_act_on_covered_episode_count = 0`. In this numerical fixture that is the mechanically checked set-inclusion implication: if the hidden pose error is inside a completion envelope whose every member passes the same numerical task reader, ACT cannot fail that same pose-error reader. This is not a physical safety certificate.

## 2x sensor-noise stress

Whole-trajectory envelope coverage: **93.33%** (56/60), Wilson 95% CI **84.07%–97.38%**.

| Task | estimator mean k | completion mean k | `k_C < k_E` | estimator completion | completion-stop completion | HOLD | unsafe ACT |
|---|---:|---:|---:|---:|---:|---:|---:|
| top suction | 13.917 | 11.300 | 95.00% | 100.00% | 98.33% | 1.67% | 0.00% |
| label alignment | 13.917 | 12.800 | 68.33% | 98.33% | 98.33% | 1.67% | 0.00% |
| keyed insertion | 13.917 | 20.000 | 0.00% | 98.33% | 0.00% | 100.00% | 0.00% |

The shifted keyed-insertion failure is intentionally retained: the completion envelope never becomes sufficiently narrow for ACT under this task reader, so the method HOLDs rather than silently transferring in-distribution usefulness to the shifted condition.

## Required interpretation

The result supports a numerical implementation claim:

```text
calibrated completion envelope
+ task-reader invariance
can produce k_C < k_E on this generated ICP fixture
```

It does **not** establish:

- real RGB-D coverage;
- FoundationPose acceleration;
- physical task non-inferiority;
- deterministic 90% safety;
- validity under arbitrary distribution shift.

## Raw-proxy ablation

The simpler raw-proxy conformal envelope in the same workflow achieved 95.83% whole-trajectory coverage on its frozen in-distribution evaluation but returned HOLD for 100% of episodes for every task. This is retained as a negative ablation:

```text
coverage alone != operationally useful completion set
```

A useful set must be both sufficiently covered and sufficiently informative for the downstream reader.
