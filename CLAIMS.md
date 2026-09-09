# Claim boundaries

| Claim | Status |
|---|---|
| Task declaration can change a stopping decision for the same unresolved pose state | Executed finite diagnostic |
| Coupled task constraints can invalidate independent-coordinate gates | Executed counterexample |
| Binary PASS/FAIL state can lose transition-relevant magnitude information | Executed counterexample |
| A finite completion set can be task-invariant without identifying one latent pose | Executed finite completion-envelope diagnostic |
| A trajectory-level split-calibration procedure can map the observable numerical ICP proxy to a frozen completion-envelope scale without giving ground truth to the online gate | Executed numerical method |
| Held-out whole-trajectory coverage can be measured independently of the adaptive stopping stage | Executed numerical method |
| Split-conformal marginal coverage is a deterministic safety guarantee | Not supported / false interpretation |
| In-distribution calibration transfers unchanged to arbitrary distribution shift | Not supported; stress condition is reported separately |
| Dijkstra and pinned IDM min-plus core recover the same modeled finite optimum in the older routing fixture | Executed finite diagnostic |
| IDM is faster than Dijkstra | Not supported |
| Coverage-qualified task stopping reduces real RGB-D/GPU 6D pose latency | Open hypothesis |
| Coverage-qualified task stopping is non-inferior on physical manipulation | Open hypothesis |
| Selective stopping is universally robust to uncertainty undercoverage | Refuted as a universal claim by the stress fixture |
| FoundationPose supports arbitrary coordinate-selective refinement unchanged | Not supported by current inspected interface |

No synthetic action cost or numerical ICP timing in this repository should be reported as a real GPU speedup, energy saving, or physical robot performance result.

The current strongest numerical question is:

```text
k_C < k_E ?
```

where `k_C` is the first refinement stage whose **calibrated completion envelope** is entirely PASS under the declared downstream task reader, and `k_E` is estimator-side convergence.
