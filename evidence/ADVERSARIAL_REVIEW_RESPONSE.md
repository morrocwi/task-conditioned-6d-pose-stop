# Response to adversarial camera-ready review of commit `0c7532d`

Review date: 9 September 2026.

This file tracks engineering changes made after the adversarial review. It is not an external peer-review response and does not convert mechanical scrutiny into external validation.

## Status table

| Review item | Status after hardening | Repository response | Remaining evidence need |
|---|---|---|---|
| R1 finite conformal rank boundary | **FIXED IN CODE** | Shared `safe_split_conformal_quantile` uses the augmented `+infinity` order statistic when rank=`n+1`; small calibration therefore becomes uninformative/HOLD. Regression tests include the `n=2, alpha=.1` counterexample. | External/statistical review still welcome. |
| R2 NaN/invalid numerics could ACT | **FIXED IN CODE** | Certificate inputs are validated; NaN/non-finite model state/overflow produces an unbounded envelope or explicit error, never a zero-width certificate. Task readers fail closed. | Hardware-specific numerical faults remain system dependent. |
| R3 degenerate non-inferiority bootstrap | **FIXED FOR LAB HARNESS** | Inferential PASS now uses paired discordances and a conservative finite-sample lower bound from Clopper–Pearson marginal bounds with Bonferroni allocation. Margin, confidence, minimum N and sampling unit are predeclared. n=1 equality does not PASS. | A real study still needs a justified sampling/power plan. |
| R4 latency accounting | **CLAIM DOWNGRADED + INTERFACE ADDED** | Prefix trajectory sums are explicitly labelled counterfactual prefix costs and cannot emit a latency PASS. `online_policy_measured` mode accepts paired separately executed policy timings including policy overhead. | Real GPU/CPU online execution is still required for a speed claim. |
| R5 weak comparator isolation | **NEW MATCHED EXPERIMENT** | `experiments/matched_baselines.py` compares full budget, default estimator stop, risk-checked fixed stage, risk-checked estimator threshold, risk-checked learned scalar score, raw envelope and trajectory-conformal certificate on the same splits. Comparator parameters are frozen on CALIBRATION before new TEST seeds are generated/evaluated. | Interpret the frozen comparison as numerical evidence only; real-backend comparisons remain open. |
| R6 repetition mismatch | **NARRATIVE FIXED** | Ten-cycle study is now labelled `[NumericalReplication-ReducedConfiguration]`; the repo explicitly states it is not a repetition of the first 160/160/120, K=20, 120-point configuration. | Same-configuration repetition can be added if needed for a specific paper claim. |
| R7 manuscript/operational semantics drift | **CODE FIXED; MANUSCRIPT/README UPDATED SEPARATELY** | No-certificate episodes now report certificate hitting time as undefined/None and terminal HOLD as a separate executed endpoint. Relative rotation-vector convention is documented. | Tables should always be generated from versioned result manifests. |
| R8 real robotics generalization | **OPEN BY DESIGN** | University lab protocol and backend-agnostic evaluator are retained; numerical results remain explicitly generated-data evidence. | Real RGB-D/pose backend; physical robot only if claiming physical manipulation. |

## Shared safety core

Correctness-critical operations are centralized in `cqts/safety.py` so the raw experiment, learned experiment, and university lab harness cannot silently drift to different conformal or inferential rules.

Mechanical regression tests cover:

- small-n conformal rank requiring `+infinity`;
- NaN calibration scores;
- invalid model coefficients and overflow;
- malformed task tolerances;
- unbounded calibration producing HOLD;
- n=1 paired equality not certifying non-inferiority;
- predeclared minimum sample size enforcement.

## Claim ceiling after revision

The strongest currently licensed claim remains numerical:

> On the declared generated-point-cloud ICP/Kabsch fixtures, a trajectory-calibrated completion set can be used as a task-specific stopping certificate and compared reproducibly against matched numerical baselines.

Still not licensed:

- real RGB-D coverage;
- FoundationPose/GPU speedup;
- physical manipulation non-inferiority;
- safety certification;
- universal OOD detection;
- universal novelty or superiority.

The post-review changes improve correctness and falsifiability. They do not manufacture external empirical evidence.
