# External Lab Result Report — run6 (PROP-NATIVE-03, memory/persistence accumulator)

## Laboratory and system

- Institution: in-house (this repository; not independent external-lab validation)
- Laboratory: task-conditioned-6d-pose-stop, sixth real-data cycle
- Date: 2026-09-09
- Backend name: in-repo numpy+scipy point-to-point ICP/Kabsch (`lab/bop_icp_backend.py`), unchanged since run4
- Backend version / git commit: predeclaration commit `efa52b88af63cc172e9d98fbaee8d74684d0e72a` (2026-09-09T16:49:07+07:00)
- Model weights identifier: none (no learned model; closed-form SVD/Kabsch, no fitted regression anywhere in the decision path)
- Camera / sensor: BOP LM-O test scene 000002 real RGB-D captures (depth only used)
- GPU / CPU: CPU only (numpy/scipy)
- Operating system: Linux (workstation)
- Object set: obj_id 1 (ape), obj_id 8 (driller) — identical to runs 1-5
- Task set: `top_suction`, `label_alignment`, `keyed_insertion` — identical tolerances to runs 2-5
- Ground-truth source: BOP LM-O `scene_gt.json`, used ONLY offline by the evaluator (never in the online decision)
- SE(3) error convention: `lab/se3.py` (translation metres, rotation vector radians)

## Mechanism

`lab/native_persistence.py` wraps run5's unmodified single-instant invariance check
(`lab/native_sensitivity.py:stage_decision`, same `lambda_ref=0.33927484832749355`,
`cap=0.03274911721483188`) in a resetting streak accumulator `M_k` (`M_k=M_{k-1}+1` if the
stage-k check passes, else `M_k=0`), gating `ACT <=> M_k >= theta`, `theta=4` (PROP-NATIVE-03).
Two predeclared variants were run:

- **variant (a)**: theta-gate only, PROP-NATIVE-02's perturbation rule reused byte-identical.
- **variant (b)**: ALSO scales the perturbation magnitude by `max(1, residual_rmse_k /
  residual_scale_ref)`, capped at `2x` run5's `CAP`, using the current online-observable ICP
  correspondence residual RMSE as a disclosed attempt to address run5's own diagnosed root cause
  (perturbation magnitude has no absolute-residual-scale term).

No ground truth is used anywhere in either variant's online decision. Oracle `abs_pose_error_6d`
is read only afterward by the evaluator, exactly the same offline-oracle role T* already plays for
runs 1-5.

## Frozen protocol

- TRAIN / CALIBRATION / FINAL TEST episodes: 40 / 40 / 40 (identical to runs 1-5)
- Split rule: identical to runs 1-5 (60 frames drawn seed 20260909, permuted seed 20260909, cut 20/20/20)
- `theta`: 4 (predeclared; see RATIONALE.md — reused from run3's own frozen Bonferroni checkpoint set `{4,8,12,15}`, not a newly invented number)
- `residual_scale_ref` (variant b only): 0.018320411722306557 (median TRAIN ICP residual RMSE)
- `residual_cap_multiplier` (variant b only): 2.0 (predeclared doubling of run5's own CAP)
- non-inferiority margin: 0.05 (reused unchanged)
- estimator-side stopping rule: identical to runs 1-5 (two consecutive stages within 0.5mm/0.1deg)
- task reader definitions: identical to run2-5, byte-identical tolerances
- Predeclaration commit `efa52b8` predates both evaluator invocations against `test.jsonl` (F4 discipline maintained)

## Honest prediction stated BEFORE opening test.jsonl (see RATIONALE.md)

TRAIN-only diagnostic (`lab/derive_native_persistence_from_train.py`, no ground truth, no
calibration/test access): PROP-NATIVE-02's single-instant check evaluates **True at every one of
640 TRAIN stage readings, for all three tasks** (initial streak length 16/16 for every TRAIN
episode). This predicted, before test.jsonl was read for run6, that variant (a) would reduce to a
deterministic fixed-stage rule (`M_k` deterministically `k+1`), not a genuine noise filter. The
test-time result below confirms that prediction exactly.

## Primary results

**Both variants produce numerically IDENTICAL results.** ACT fires deterministically at
`k=theta-1=3` on 100% of test episodes, for all three tasks, in both variants — variant (b)'s
residual-scale multiplier never changed a single ACT/CONTINUE decision (see "Why variant (b) made
no difference" below).

| Task | ACT rate | k at ACT | estimator completion | ACT-endpoint completion | HOLD | unsafe ACT | non-inferiority |
|---|---:|---:|---:|---:|---:|---:|---|
| top_suction | 100% | 3 (fixed) | 32.5% | 20.0% | 0% | 32/40 (80.0%) | FAILS (LCB -35.7%) |
| label_alignment | 100% | 3 (fixed) | 27.5% | 20.0% | 0% | 32/40 (80.0%) | FAILS (LCB -31.5%) |
| keyed_insertion | 100% | 3 (fixed) | 10.0% | 12.5% | 0% | 35/40 (87.5%) | FAILS (LCB -17.6%) |

(Identical in both variant (a) and variant (b) — see `lab_results_variant_a.json` and
`lab_results_variant_b.json` for the full byte-for-byte payloads; the two files differ only in
`protocol.variant`, `protocol.decision_rule` text, and per-stage `magnitude`/`residual_scale_
applied` audit fields, never in any ACT/HOLD/completion/unsafe outcome.)

## Mandatory failure accounting

- Number of unsafe ACTs total: 32/40 (top_suction), 32/40 (label_alignment), 35/40 (keyed_insertion) — **99 unsafe ACTs out of 120 task-episode pairs (82.5%)**
- Number of HOLD episodes: 0 for all three tasks, both variants
- Cases where `k_ACT >= k_estimator`: k=3 is before the estimator's own convergence stage on the
  large majority of episodes (estimator completion rates 10.0-32.5% show most episodes have not
  yet converged by k=3 either — this is not evidence the ACT stage is "late enough", only that
  neither the ACT stage nor the estimator's own stopping rule has converged the pose by then)
- `degenerate_h_k_stage_rate`: 0.0 for all three tasks, both variants — H_k was never numerically
  singular; the result is the ordinary (non-degenerate) branch of the rule, same as run5

## Comparison against run5 (PROP-NATIVE-02, single-instant, no memory)

| Metric | run5 (theta=1, k=0) | run6 variant a/b (theta=4, k=3, fixed) |
|---|---:|---:|
| ACT rate | 100% | 100% |
| unsafe ACT rate (mean of 3 tasks) | 95.0% | 82.5% |
| top_suction unsafe | 92.5% | 80.0% |
| label_alignment unsafe | 100% | 80.0% |
| keyed_insertion unsafe | 92.5% | 87.5% |

Adding four stages of persistence (delaying the fixed ACT stage from k=0 to k=3) reduced the
unsafe-ACT rate modestly (95.0% -> 82.5%, a ~13-point drop) simply because a few more ICP
refinement iterations ran before the estimate was locked in — the SAME improvement any fixed
`k=3` early-stopping rule with no invariance check at all would show, since `M_k` never actually
varied by episode (see next section). **This is not evidence that persistence filters noise or
makes the construction safe** — it is evidence that running 3 more ICP iterations before
committing helps a little, which was already known independent of this mechanism.

## Root-cause diagnosis (still unsafe)

**Persistence alone (variant a) does not fix run5's diagnosed problem, exactly as predicted before
this run started.** The underlying single-instant check (PROP-NATIVE-02) is not noisy on this
backend/dataset — it is a near-constant function of stage (True at essentially every stage for
essentially every episode, confirmed both on TRAIN before this run and on the FINAL TEST data
itself: every episode's `M_k` sequence is `[1,2,3,4,...]` with no reset ever observed). A streak
counter has nothing to filter when the underlying signal never toggles. `theta` therefore acts
purely as a **fixed-delay knob** (`ACT` always at `k=theta-1`), structurally indistinguishable from
a naive "always refine for exactly `theta` stages then commit" rule that consults neither `H_k`
nor the task reader's invariance property at all. The improvement observed (95.0% -> 82.5% unsafe)
is consistent with that reading: it is the ordinary benefit of 3 extra ICP iterations, not a
safety property of the persistence mechanism.

**Variant (b) (residual-scaled magnitude) also did not help, for a distinct, disclosed reason**: at
`k=0` the residual-scale multiplier reached only `~1.15x` (see `lab_results_variant_b.json`'s
per-episode audit; sampled directly during this write-up, episode 0 stage 0: `magnitude_base=
0.0327`, `magnitude=0.0378`, `residual_scale_applied=1.15`), and the `2x` cap (`0.0655`) is still
roughly two orders of magnitude smaller than the real coarse-detector initial-pose error (isotropic
translation noise sigma=15mm, rotation error Uniform[5,20]deg per `lab/bop_icp_backend.py:
perturb_pose`). The ICP correspondence residual RMSE (the observable this variant scales by) is a
local point-to-plane fit quality statistic, order 15-30mm, and its ratio to its own TRAIN-typical
value varies only by roughly 1-2x across a trajectory — it simply does not carry enough dynamic
range to bridge a two-order-of-magnitude gap to the coarse detector's actual error scale. This
confirms, on a second, independently-designed construction, run5's own root-cause finding: no
observable quantity exposed by this ICP backend to date (H_k's spectral floor, or the
correspondence residual RMSE) captures the ABSOLUTE scale of the pose estimate's real error
relative to ground truth — both are LOCAL statistics about the current correspondence set's
conditioning/fit quality, not proxies for how far the current estimate actually is from correct.

## What this does and does not establish

**Does NOT establish:** that PROP-NATIVE-03 (memory/persistence) is a viable fix for PROP-NATIVE-02's
unsafe-ACT finding — it is refuted, in both the plain (a) and residual-scaled (b) forms tested here.
It does NOT establish that persistence/memory constructions are never useful for this problem in
principle — only that adding persistence on top of a NON-NOISY (near-constant) single-instant
signal cannot do anything but act as a fixed-delay knob, and that widening the observable-scaled
magnitude term by the specific residual-RMSE-based factor tried here is nowhere near large enough
to matter.

**Does establish:** (1) PROP-NATIVE-03, implemented per its own registered/disclosed streak
semantics, is REFUTED as a fix for run5's unsafe-ACT finding on this backend and dataset — the
predicted structural failure mode (fixed-delay reduction, not noise filtering) was stated before
running against test and is confirmed by the test-time numbers. (2) The specific residual-RMSE-based
magnitude-scaling variant tried here (variant b) is also refuted, and the reason is now precisely
located: the observable residual signal's dynamic range (~1-2x) is far too small relative to the gap
between the current perturbation cap and the real initial-pose-error scale (~10-100x) for any
bounded multiplicative correction of this kind to close it. (3) Any future construction along this
line needs either an observable with genuinely large dynamic range tied to absolute error scale (not
yet identified in this backend's exposed features), or must abandon the "perturb-and-check-invariance
around the CURRENT estimate" framing entirely in favor of a mechanism that reasons about the
INITIAL/coarse estimate's own error distribution directly.

## Novelty/positioning (unchanged guardrails)

This result does not "solve 6D pose estimation" and is not claimed as first anything. Consistent
with `ops/HANDOFF_2026-09-09_external_dataset.md`'s positioning constraint: "We are not trying to
eliminate perception error entirely. We ask whether the error that remains still affects what the
robot is about to do." This run's honest answer, for both tested constructions, is: neither
correctly answers that question at this trajectory stage — persistence alone converts an
already-vacuous single-instant check into a fixed-delay rule, and residual-scaling the magnitude by
the one available factor tried here does not close the gap to the real error scale.

## Claim form

On the in-repo numpy+scipy ICP backend / BOP LM-O real depth data / CPU, PROP-NATIVE-03's
memory/persistence accumulator (`theta=4`, gating run5's unmodified single-instant invariance
check) licensed ACT deterministically at the fixed stage `k=3` on 100% of held-out trials, across
all three declared tasks, and that ACT was incorrect (unsafe) on 80.0-87.5% of those trials — a
modest improvement over run5's 92.5-100% unsafe rate, attributable to the extra ICP iterations run
before commit, not to any noise-filtering property of the persistence mechanism. A disclosed
residual-scaled variant (b) produced byte-identical outcomes to variant (a), because its magnitude
correction is roughly two orders of magnitude too small to matter. Both variants are reported as
refuted, safety-relevant negative results, not positive certificate-works claims.

## Physical robot extension

Not applicable — no physical manipulation was executed. Pose-ground-truth replay only.
