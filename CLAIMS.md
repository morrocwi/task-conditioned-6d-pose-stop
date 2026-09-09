# Claim boundaries

| Claim | Status |
|---|---|
| Task declaration can change a stopping decision for the same unresolved pose state | Executed finite diagnostic |
| Coupled task constraints can invalidate independent-coordinate gates | Executed counterexample |
| Binary PASS/FAIL state can lose transition-relevant magnitude information | Executed counterexample |
| A finite completion set can be task-invariant without identifying one latent pose | Executed finite completion-envelope diagnostic |
| A trajectory-level split-calibration procedure can map observable numerical refinement features to a frozen completion-envelope scale without giving ground truth to the online gate | Executed numerical method |
| Held-out whole-trajectory coverage can be measured independently of the adaptive stopping stage | Executed numerical method |
| A raw-proxy high-coverage envelope is necessarily operationally useful | Refuted in the frozen ablation: 95.83% coverage, 100% HOLD |
| The first frozen learned final test empirically achieved the nominal 90% trajectory coverage target | Not supported: observed 85.00%, Wilson 95% CI 77.53–90.30% |
| The frozen learned procedure can repeatedly realize coverage near the nominal target within the same declared numerical generator | Supported by 10-cycle numerical replication: mean 91.17%, median 91.67%, range 83.33–95.00%, 9/10 cycles >=90% |
| The frozen learned procedure repeatedly yields `k_C < k_E` on the numerical backend | Supported in 10-cycle replication: mean 92.33% suction, 91.33% label, 83.83% insertion |
| Completion-stop task completion remains close to estimator-stop completion in the repeated numerical study | Supported numerically: mean differences -1.17 pp, 0.00 pp, -0.33 pp by task |
| The reported numerical studies contain unsafe ACT on a trajectory that is actually inside a robust-PASS completion envelope | Not observed; mechanical count = 0 for implemented readers |
| Split-conformal marginal coverage is a deterministic safety guarantee | Not supported / false interpretation |
| In-distribution calibration transfers unchanged to arbitrary distribution shift | Not supported; shifted conditions are reported separately |
| Dijkstra and pinned IDM min-plus core recover the same modeled finite optimum in the older routing fixture | Executed finite diagnostic |
| IDM is faster than Dijkstra | Not supported |
| Coverage-qualified task stopping reduces real RGB-D/GPU 6D pose latency | Open hypothesis |
| Coverage-qualified task stopping is non-inferior on physical manipulation | Open hypothesis |
| A real numpy+scipy ICP backend on real BOP LM-O RGB-D data can produce a real-sensor completion envelope via the shared `cqts/safety.py` split-conformal machinery | Executed real-sensor method (`lab/results/real-bop-lmo-2026-09-09/`); envelope was finite (not the +infinity fail-closed case) |
| That real-sensor envelope is task-discriminative enough to yield `k_C < k_E` on real BOP LM-O data at the reused numerical-fixture tolerances | Refuted for this run: 0/40 test episodes for all three declared tasks (100% HOLD, `certificate_rate = 0.0`); see falsifier accounting in `lab/results/real-bop-lmo-2026-09-09/RESULT.md` |
| Real-sensor coverage-qualified stopping is non-inferior to estimator-side stopping on real BOP LM-O data at this scope | Not supported: non-inferiority FAILS its predeclared 5% margin for all three tasks (lower confidence bound -13.2% / -13.2% / -8.8%) |
| That real-sensor envelope's non-discriminativeness on real BOP LM-O data is caused by task tolerances too tight for this backend's real achievable accuracy | Refuted by the second run (`lab/results/real-bop-lmo-2026-09-09-run2/`): tolerances re-derived from TRAIN-split achievable ICP accuracy (loosened 2.6-2.8x translation, 1.4-1.8x rotation, raising the estimator's own completion rate 2.5%->32.5% / 2.5%->27.5% / 0%->10%) still produced `certificate_rate = 0.0` and 0/40 `k_C<k_E` for all three tasks; the calibrated envelope width (`q≈2.0`), not the tolerance value, is the more likely bottleneck at this scope |
| Real-sensor coverage-qualified stopping is non-inferior to estimator-side stopping on real BOP LM-O data with tolerances re-derived from TRAIN-split achievable accuracy | Not supported, more clearly rejected than the first run: non-inferiority FAILS its predeclared 5% margin for all three tasks (lower confidence bound -49.1% / -43.9% / -23.7%), see `lab/results/real-bop-lmo-2026-09-09-run2/RESULT.md` |
| A Bonferroni-corrected multi-checkpoint conformal band (K'=4 predeclared checkpoints, coordinate-max-only nonconformity, PROP-CONF-03) produces per-checkpoint quantiles smaller than the joint whole-trajectory q≈2.0008, raising the certificate rate above 0% on the same real BOP LM-O data | Refuted (`lab/results/real-bop-lmo-2026-09-09-run3/RESULT.md`): per-checkpoint quantiles came out 4.3-10.0% LARGER (q in [2.085, 2.200]), not smaller, and certificate rate stayed at exactly 0.0 for all three tasks (100% HOLD), identical in kind to runs 1-2 |
| The PROP-CONF-03 union-bound construction's own coverage guarantee (>=90% whole-trajectory coverage via K' independently-calibrated checkpoints) held on real BOP LM-O data at n=40 | Supported: all-checkpoints-covered rate 97.5% (Wilson 95% CI 87.1-99.6%), consistent with (not a violation of) the 90% target -- see `lab/results/real-bop-lmo-2026-09-09-run3/RESULT.md` |
| Real-sensor coverage-qualified stopping under the PROP-CONF-03 multi-checkpoint construction is non-inferior to estimator-side stopping on real BOP LM-O data with run2's TRAIN-derived tolerances | Not supported: non-inferiority FAILS its predeclared 5% margin for all three tasks, identical lower confidence bounds to run 2 (-49.1% / -43.9% / -23.7%), because the certificate policy's completion outcomes are mechanically identical to run 2 whenever certificate_rate=0.0 under both constructions |
| Selective stopping is universally robust to uncertainty undercoverage | Refuted as a universal claim by the stress fixture |
| FoundationPose supports arbitrary coordinate-selective refinement unchanged | Not supported by current inspected interface |

No synthetic action cost or numerical ICP timing in this repository should be reported as a real GPU speedup, energy saving, or physical robot performance result.

The current strongest numerical statement is not merely that `k_C < k_E` occurred once. The frozen learned procedure now has:

```text
first frozen final test
+ 10 repeated TRAIN -> CALIBRATION -> TEST cycles
+ cross-platform CI
+ explicit negative ablations
```

within the same declared numerical generator.

The real-backend question remains:

```text
Can a real iterative 6D perception backend construct a sufficiently covered,
task-discriminative completion envelope early enough that k_C < k_E reduces
end-to-end latency without unacceptable physical task-completion loss?
```
