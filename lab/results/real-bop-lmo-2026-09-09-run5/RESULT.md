# External Lab Result Report — run5 (PROP-NATIVE-01/02, H3, native retained-sensitivity model)

## Laboratory and system

- Institution: in-house (this repository; not independent external-lab validation)
- Laboratory: task-conditioned-6d-pose-stop, fifth real-data cycle
- Date: 2026-09-09
- Backend name: in-repo numpy+scipy point-to-point ICP/Kabsch (`lab/bop_icp_backend.py`), unchanged since run4
- Backend version / git commit: predeclaration commit `43954c09eedda7f3802adee6d83b559c836d19e6` (2026-09-09T16:03:50+07:00)
- Model weights identifier: none (no learned model; closed-form SVD/Kabsch, and no fitted regression of any kind in the decision path itself — see "Mechanism" below)
- Camera / sensor: BOP LM-O test scene 000002 real RGB-D captures (depth only used)
- GPU / CPU: CPU only (numpy/scipy)
- Operating system: Linux (workstation)
- Object set: obj_id 1 (ape), obj_id 8 (driller) — identical to runs 1-4
- Task set: `top_suction`, `label_alignment`, `keyed_insertion` — identical tolerances to runs 2-4
- Ground-truth source: BOP LM-O `scene_gt.json`, used ONLY offline by the evaluator (never in the online decision)
- SE(3) error convention: `lab/se3.py` (translation metres, rotation vector radians)

## Mechanism (genuinely different from runs 1-4 — read this before the table)

No fitted error-shape model, no calibrated completion set, no conformal quantile anywhere in this
run. At each ICP stage k, `lab/native_sensitivity.py` takes the real ICP normal-equations Hessian
`H_k = J^T J`, finds its single smallest eigenvalue/eigenvector `(lambda_min, v_min)`, perturbs the
current pose estimate `T_hat_k` by `+-m*v_min` with `m = min(C/lambda_min, CAP)` (`lambda_ref` and
`CAP` both derived from TRAIN only — see RATIONALE.md), and ACTs iff
`cqts.safety.fail_closed_task_pass` gives the SAME verdict for the unperturbed pose and both
perturbed candidates. **No ground truth is used anywhere in this online decision.** Oracle
`abs_pose_error_6d` is read only afterward, by the evaluator, exactly the same offline-oracle role
T* already plays for runs 1-4.

## Frozen protocol

- TRAIN episodes: 40
- CALIBRATION episodes: 40
- FINAL TEST episodes: 40
- Split rule: identical to runs 1-4 (60 frames drawn seed 20260909, permuted seed 20260909, cut 20/20/20)
- `alpha`: not applicable (no calibrated quantile in this mechanism)
- nominal whole-trajectory coverage: not applicable
- non-inferiority margin: 0.05 (reused unchanged)
- estimator-side stopping rule: identical to runs 1-4 (two consecutive stages within 0.5mm/0.1deg)
- online feature vector definition: identical `observable_features`, plus the additive `native` field (H_k flattened, T_hat_k)
- task reader definition: identical `top_suction`/`label_alignment`/`keyed_insertion` box/l1 specs, byte-identical tolerances from run2-4
- deviations from `lab/README.md`: this run does not use `lab/run_real_system.py`'s calibration machinery at all; see `lab/run_native_sensitivity.py`

## Primary results

**IMPORTANT: this construction licensed ACT on every test episode, and ACT was WRONG on the large
majority of them. This is a genuine, serious safety finding — see "Mandatory failure accounting"
and "What this does and does not establish" below before reading the table as if it were a success.**

| Task | ACT rate | mean k at ACT | estimator completion | ACT-endpoint completion | HOLD | unsafe ACT | non-inferiority |
|---|---:|---:|---:|---:|---:|---:|---|
| top_suction | 100% | 0.0 | 32.5% | 7.5% | 0% | 37/40 (92.5%) | FAILS (LCB -47.6%) |
| label_alignment | 100% | 0.0 | 27.5% | 0.0% | 0% | 40/40 (100%) | FAILS (LCB -43.9%) |
| keyed_insertion | 100% | 0.0 | 10.0% | 7.5% | 0% | 37/40 (92.5%) | FAILS (LCB -19.8%) |

`k=0` is the very FIRST ICP stage — the perturbed, un-refined initial pose (the simulated coarse
detector output). ACT fires immediately, before any refinement iteration, on every single test
episode, for every task.

## Mandatory failure accounting

- Number of unsafe ACTs on covered episodes: not applicable (no calibrated coverage concept in this mechanism)
- Number of unsafe ACTs total: 37/40 (top_suction), 40/40 (label_alignment), 37/40 (keyed_insertion) — **114 unsafe ACTs out of 120 task-episode pairs (95.0%)**
- Number of HOLD episodes: 0 for all three tasks
- Cases where `k_ACT >= k_estimator`: 0/40 for all three tasks (ACT always precedes the estimator's own convergence stage, since it fires at k=0)
- Calibration/test coverage shortfall, if any: not applicable
- Distribution-shift conditions tested: none (identical data/split to runs 1-4)
- `degenerate_h_k_stage_rate` (fraction of stages where H_k was numerically singular and the
  fail-closed CAP branch fired): 0.0 for all three tasks — the invariance test never hit the
  degenerate-H_k fail-closed path; the near-universal wrong ACT is not an artifact of the
  degenerate-H_k handling, it is the ordinary (non-degenerate) branch of the rule doing this.

## Root cause (diagnosed, not merely observed)

The perturbation magnitude `m` this construction produces (bounded by `CAP≈0.033`, itself already
capped well below the task tolerances) is systematically far SMALLER than the REAL pose error at
`k=0`: the simulated coarse-detector initial pose carries translation noise sigma=15mm and rotation
error Uniform[5,20]deg (`lab/bop_icp_backend.py:perturb_pose`), routinely tens of millimetres/degrees
off ground truth — an order of magnitude or more larger than `m`. Because the invariance test only
asks "does the task verdict change within a small neighbourhood of the CURRENT estimate", and that
neighbourhood is far smaller than the estimate's actual (unknown, ground-truth-relative) error, the
test is answering a different, much easier question than "is the current estimate good enough" — it
is answering "is a tiny wobble around wherever I currently am tolerable", which is close to always
true regardless of how far off the current estimate actually is. `H_k`'s spectral floor
(`lambda_min`, order 0.12-1.08 across TRAIN, never degenerate) measures LOCAL geometric
conditioning of the correspondence set, not the ABSOLUTE scale of the estimate's own residual error
relative to ground truth — those are different quantities, and this run is the direct empirical
demonstration that conflating them (perturbation magnitude derived from `1/lambda_min` alone, with
no residual/noise-scale term) produces a dangerously overconfident stopping rule. This is exactly
the assumption PROP-NATIVE-02's own `honest_caveats` flagged as unverified ("the relationship
between 'small eigenvalue direction' and 'actual pose error along that direction' is itself an
assumption ... must be checked empirically") — **this run refutes that assumption empirically, at
least at this trajectory stage and this backend.**

Contrast with `lab/decay_predictor.py`'s own predictor (run4, PROP-DECAY-01): that construction
multiplied its contraction-rate term by the CURRENT observable residual proxy magnitude
(`s_k,i = rho_k^(K-k) * |proxy_i|`), i.e. it did carry an absolute-scale term. This run's mechanism,
as predeclared per PROP-NATIVE-02's literal statement (perturbation magnitude proportional to
`1/lambda_j`, bounded by a cap), deliberately does NOT multiply by any residual/noise-scale
observable — that omission is exactly what this result exposes as unsafe.

## What this does and does not establish

**Does NOT establish:** that H3/PROP-NATIVE-01/02 is a viable stopping certificate on this backend
at this stage of refinement. It licenses ACT far too readily and is wrong 95% of the time when it
does. It also does NOT establish that PROP-NATIVE-01's underlying critique (T* is a non-readout,
should not enter even offline calibration) is wrong — this run only tested PROP-NATIVE-02's specific
*use* of that idea (spectral-floor-only perturbation magnitude), not the broader native-error
framing.

**Does establish:** (1) this specific construction, implemented exactly as predeclared from
PROP-NATIVE-02's own statement, produces a dangerously overconfident stopping rule at the very first
ICP stage on this real backend and dataset — the single most safety-relevant finding of any of the
five real-data cycles to date, because runs 1-4 all failed SAFE (100% HOLD, never an unsafe ACT);
this is the first cycle to produce an ACT-licensing mechanism at all, and it is unsafe when it does.
(2) `H_k`'s spectral floor alone, without pairing to an absolute residual/noise-scale observable, is
not a safe proxy for pose-estimate uncertainty on this backend — any future construction reusing
`H_k`'s eigenstructure for a stopping decision should combine it with a residual-magnitude term
(as run4's own decay predictor already did, for a different purpose), not use it standalone. (3) the
degenerate-`H_k` fail-closed branch (CAP substitution on a singular Hessian) was never exercised here
(rate 0.0), so this failure mode is not an artifact of that particular guard.

## Novelty/positioning (unchanged guardrails)

This result does not "solve 6D pose estimation" and is not claimed as first anything. Consistent
with `ops/HANDOFF_2026-09-09_external_dataset.md`'s positioning constraint: "We are not trying to
eliminate perception error entirely. We ask whether the error that remains still affects what the
robot is about to do." This run's honest answer, for THIS specific construction, is: the construction
as built does not correctly answer that question at k=0 — it answers a much narrower, misleadingly
easy question instead (see "Root cause" above).

## Claim form

On the in-repo numpy+scipy ICP backend / BOP LM-O real depth data / CPU, the native
retained-sensitivity construction (PROP-NATIVE-02, H_k spectral-floor perturbation, no ground truth
in the online decision) licensed ACT at the very first refinement stage on 100% of held-out trials,
across all three declared tasks, and that ACT was incorrect (unsafe) on 92.5-100% of those trials.
This is reported as a safety-relevant negative result, not a positive certificate-works claim.

## Physical robot extension

Not applicable — no physical manipulation was executed. Pose-ground-truth replay only.
