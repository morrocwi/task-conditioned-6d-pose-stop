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
