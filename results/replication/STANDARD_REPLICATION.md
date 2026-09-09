# Frozen repeated TRAIN–CALIBRATION–TEST replication study

**Evidence label:** `[NumericalReplication]`  
**GitHub Actions run:** `34303076954`  
**Evaluated commit:** `0469be38edca9e5dadba3791b6ccdb95ebe63d9b`  
**Standard profile:** 10 independent numerical replication cycles

This study does not tune the learned completion method. Each replicate reruns the complete frozen procedure with new deterministic seeds:

```text
TRAIN -> CALIBRATION -> TEST
```

within the same declared generated-point-cloud ICP/Kabsch data-generating process.

Per replicate:

- TRAIN: 80 episodes
- CALIBRATION: 80 episodes
- TEST: 60 episodes
- maximum refinement stage: 16
- points per cloud: 80
- nominal marginal whole-trajectory coverage: 90%

## Coverage across independent cycles

| Statistic | Realized whole-trajectory coverage |
|---|---:|
| Mean | 91.17% |
| Median | 91.67% |
| Minimum | 83.33% |
| Maximum | 95.00% |
| Replicates at or above nominal 90% | 9 / 10 |

The earlier first frozen 120-episode learned final test observed 85.00% coverage. These repeated cycles show that one realized test proportion should not be treated as the conformal theorem or as the stable empirical coverage of the procedure. Within this declared numerical generator, the frozen procedure's mean realized coverage across these 10 cycles was close to and slightly above the nominal target, while individual cycles still varied substantially.

This is a numerical replication diagnostic, not a proof that all future datasets or real sensors attain 90% coverage.

## Task-stopping utility across cycles

| Task | mean `k_C < k_E` | mean `k_C-k_E` | mean completion difference vs estimator stop | mean HOLD | mean unsafe ACT |
|---|---:|---:|---:|---:|---:|
| top suction | 92.33% | -3.228 | -1.17 pp | 6.00% | 0.00% |
| label alignment | 91.33% | -2.770 | 0.00 pp | 6.17% | 0.00% |
| keyed insertion | 83.83% | -1.632 | -0.33 pp | 6.50% | 0.00% |

Across all 10 cycles and all three implemented task readers:

```text
total unsafe ACT on covered episodes = 0
```

That zero is a mechanical consequence/diagnostic of the matched numerical containment and task-reader construction, not a physical safety result.

## Interpretation

The combination of the first frozen final test and this repeated-cycle study supports a stronger but still bounded statement:

```text
Within the declared numerical generator,
the frozen learned-shape + trajectory-conformal procedure
can repeatedly construct completion envelopes that are
both sufficiently informative for task stopping and approximately
aligned with the nominal marginal coverage target.
```

The study also preserves two important non-collapses:

```text
nominal conformal coverage != every realized test proportion
numerical replication != real RGB-D / physical robot replication
```

The next evidence step is therefore a real iterative 6D perception backend with independent calibration and outcome measurement, not further retuning on this numerical generator.
