# Numerical ICP experiment

> **[NumericalBackend]** Generated 3-D point clouds + iterative ICP/Kabsch were executed. No camera, neural pose model, contact physics, or physical robot was executed.

Frozen standard profile: 96 tuning episodes, 96 disjoint safety-calibration episodes, 120 held-out in-distribution episodes (3 test seeds), 60 held-out stress episodes, 20 maximum ICP iterations, 120 points/object.

The task gate uses an observable local dispersion proxy. A threshold is tuned on one split, frozen, checked on a disjoint safety-calibration split, then evaluated on held-out test seeds. Ground-truth pose error is never passed to the gate.

## Frozen task thresholds

| Task | threshold | safety-cal act | safety-cal unsafe ACT | unsafe 95% CI |
|---|---:|---:|---:|---:|
| top_suction | 0.588633 | 90.62% | 1.04% | [0.18%, 5.67%] |
| label_alignment | 0.513379 | 89.58% | 1.04% | [0.18%, 5.67%] |
| keyed_insertion | 0.758022 | 88.54% | 0.00% | [0.00%, 3.85%] |

## Held-out: in_distribution

### top_suction

| Policy | mean k | completion | HOLD | unsafe ACT | mean latency ms |
|---|---:|---:|---:|---:|---:|
| fixed8 | 8.000 | 62.50% | 0.00% | 37.50% | 7.742 |
| full | 20.000 | 94.17% | 0.00% | 5.83% | 17.910 |
| estimator_stop | 14.033 | 94.17% | 0.00% | 5.83% | 12.877 |
| task_stop | 10.292 | 93.33% | 6.67% | 0.00% | 9.710 |

Paired `k_task-k_estimator`: **-3.742**, bootstrap 95% CI **[-4.042, -3.367]**.  
`k_task < k_estimator`: **92.50%**.  
Completion difference (task - estimator): **-0.83 percentage points**.

### label_alignment

| Policy | mean k | completion | HOLD | unsafe ACT | mean latency ms |
|---|---:|---:|---:|---:|---:|
| fixed8 | 8.000 | 47.50% | 0.00% | 52.50% | 7.742 |
| full | 20.000 | 93.33% | 0.00% | 6.67% | 17.910 |
| estimator_stop | 14.033 | 93.33% | 0.00% | 6.67% | 12.877 |
| task_stop | 10.550 | 91.67% | 6.67% | 1.67% | 9.928 |

Paired `k_task-k_estimator`: **-3.483**, bootstrap 95% CI **[-3.775, -3.133]**.  
`k_task < k_estimator`: **90.83%**.  
Completion difference (task - estimator): **-1.67 percentage points**.

### keyed_insertion

| Policy | mean k | completion | HOLD | unsafe ACT | mean latency ms |
|---|---:|---:|---:|---:|---:|
| fixed8 | 8.000 | 38.33% | 0.00% | 61.67% | 7.742 |
| full | 20.000 | 92.50% | 0.00% | 7.50% | 17.910 |
| estimator_stop | 14.033 | 92.50% | 0.00% | 7.50% | 12.877 |
| task_stop | 11.542 | 90.83% | 9.17% | 0.00% | 10.761 |

Paired `k_task-k_estimator`: **-2.492**, bootstrap 95% CI **[-2.742, -2.167]**.  
`k_task < k_estimator`: **90.00%**.  
Completion difference (task - estimator): **-1.67 percentage points**.

## Held-out: stress_2x_sensor_noise

The thresholds above are frozen; they are not re-tuned under stress.

| Task | estimator completion | task completion | task HOLD | task unsafe ACT | `k_task < k_estimator` |
|---|---:|---:|---:|---:|---:|
| top_suction | 96.67% | 95.00% | 5.00% | 0.00% | 91.67% |
| label_alignment | 96.67% | 70.00% | 30.00% | 0.00% | 70.00% |
| keyed_insertion | 95.00% | 0.00% | 100.00% | 0.00% | 0.00% |

The stress result is part of the claim boundary, not an excluded failure case. Under this shift, selectivity can lose operational value by becoming too conservative.

## Evidence ceiling

These results support only the behavior of the public numerical registration fixture under its declared generator and split protocol. Wall-clock time is machine-specific. The standard profile took about 20.88 s on the author's execution environment, but other machines are expected to differ.

RGB-D/GPU/robot speedup and physical manipulation non-inferiority remain **HOLD / open hypotheses**.
