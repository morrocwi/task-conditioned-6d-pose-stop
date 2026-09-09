# Frozen same-split matched comparator result

Label: `[MatchedComparison-NewTestSeeds]`  
Execution: GitHub Actions standard-evidence job  
Workflow run: `34312608667`  
Reviewed code commit for this run: `990f49a038415e1f25c49324a1693d3f0253beaf`  
Artifact: `completion-standard-ubuntu` (artifact ID `10088946864`)

This is generated-point-cloud ICP/Kabsch evidence. It is not a camera, neural pose, GPU-speed, or physical-robot result.

## Frozen design

```text
TRAIN        160 episodes
CALIBRATION  160 episodes
TEST         120 episodes (3 new comparison seeds)
max stage     20
points       120
risk target  10%
risk confidence 95%
```

The comparison TEST seeds were distinct from the earlier learned final-test seeds. Fixed-stage, estimator-threshold, and learned-scalar comparator parameters were chosen from CALIBRATION only before TEST trajectories were generated/evaluated in the script. No TEST-outcome retuning was performed.

All policies use the same generated TEST trajectories and task readers.

## Top suction

| Policy | mean endpoint k | completion | HOLD | unsafe ACT | mean trajectory-prefix ms* |
|---|---:|---:|---:|---:|---:|
| Full budget | 20.000 | 97.50% | 0.00% | 2.50% | 21.60 |
| Estimator default | 13.100 | 97.50% | 0.00% | 2.50% | 14.56 |
| Fixed, risk-checked | 18.000 | 95.83% | 0.00% | 4.17% | 19.62 |
| Estimator threshold, risk-checked | 11.300 | 95.00% | 0.00% | 5.00% | 12.71 |
| Learned scalar, risk-checked | **8.283** | 95.00% | 2.50% | 2.50% | 9.61 |
| Raw completion envelope | 20.000 | 0.00% | 100.00% | 0.00% | 21.60 |
| **Trajectory-conformal completion** | 9.708 | **96.67%** | 3.33% | **0.00%** | 11.08 |

## Label alignment

| Policy | mean endpoint k | completion | HOLD | unsafe ACT | mean trajectory-prefix ms* |
|---|---:|---:|---:|---:|---:|
| Full budget | 20.000 | 95.83% | 0.00% | 4.17% | 21.60 |
| Estimator default | 13.100 | 95.83% | 0.00% | 4.17% | 14.56 |
| Fixed, risk-checked | 20.000 | 95.83% | 0.00% | 4.17% | 21.60 |
| Estimator threshold, risk-checked | 13.100 | 95.83% | 0.00% | 4.17% | 14.56 |
| Learned scalar, risk-checked | **9.025** | 94.17% | 2.50% | 3.33% | 10.37 |
| Raw completion envelope | 20.000 | 0.00% | 100.00% | 0.00% | 21.60 |
| **Trajectory-conformal completion** | 10.167 | **95.83%** | 3.33% | **0.83%** | 11.55 |

## Keyed insertion

| Policy | mean endpoint k | completion | HOLD | unsafe ACT | mean trajectory-prefix ms* |
|---|---:|---:|---:|---:|---:|
| Full budget | 20.000 | 95.00% | 0.00% | 5.00% | 21.60 |
| Estimator default | 13.100 | 95.00% | 0.00% | 5.00% | 14.56 |
| Fixed, risk-checked | 20.000 | 95.00% | 0.00% | 5.00% | 21.60 |
| Estimator threshold, risk-checked | 13.100 | 95.00% | 0.00% | 5.00% | 14.56 |
| Learned scalar, risk-checked | **9.800** | 92.50% | 3.33% | 4.17% | 11.17 |
| Raw completion envelope | 20.000 | 0.00% | 100.00% | 0.00% | 21.60 |
| **Trajectory-conformal completion** | 11.025 | **95.00%** | 5.00% | **0.00%** | 12.43 |

`*` Trajectory-prefix times are counterfactual sums from complete generated trajectories. They are not direct separately executed online-policy speed measurements.

## What this comparison supports

The matched result rejects a simplistic claim that the certificate is always the earliest stopping method. The learned scalar policy stops earlier on all three numerical readers.

The more specific observation is a trade-off:

- the scalar comparator stops earlier but has more unsafe ACTs and, for alignment/insertion, lower overall completion;
- the trajectory-conformal completion policy is more conservative, with later endpoints than the scalar comparator but lower observed unsafe ACT rates;
- for keyed insertion, the completion policy matches estimator-default completion (95%) while converting the estimator's 5% unsafe failures into 5% HOLD and using a lower mean endpoint (11.025 versus 13.100);
- the raw high-coverage envelope remains operationally useless at 100% HOLD.

This is evidence for a **coverage/risk-qualified stopping trade-off**, not evidence that set-valued certification is universally faster or superior.

## Statistical caution

Observed zero unsafe ACTs do not establish zero population risk. For 0/120 events, the two-sided Wilson 95% upper endpoint is about 3.10% per assigned episode. The matched experiment was designed primarily to discriminate numerical policy behavior; population-level non-inferiority and real-backend latency require the predeclared lab protocol.
