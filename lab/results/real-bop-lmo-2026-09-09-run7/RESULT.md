# External Lab Result Report — run7 (PROP-NATIVE-04, conformal-scaled Keystone-energy decay accumulator)

## Laboratory and system

- Institution: in-house (this repository; not independent external-lab validation)
- Laboratory: task-conditioned-6d-pose-stop, seventh real-data cycle
- Date: 2026-09-09
- Backend name: in-repo numpy+scipy point-to-point ICP/Kabsch (`lab/bop_icp_backend.py`), unchanged since run4
- Backend version / git commit: predeclaration commit `a76a8b9` (frozen before `test.jsonl` was opened)
- Model weights identifier: log-linear regression (`lab.run_real_system.fit_model`, unchanged since run1), no learned neural model
- Camera / sensor: BOP LM-O test scene 000002 real RGB-D captures (depth only used)
- GPU / CPU: CPU only (numpy/scipy)
- Operating system: Linux (workstation)
- Object set: obj_id 1 (ape), obj_id 8 (driller) — identical to runs 1-6
- Task set: `top_suction`, `label_alignment`, `keyed_insertion` — identical tolerances to runs 2-6
- Ground-truth source: BOP LM-O `scene_gt.json`, used ONLY offline by the evaluator (never in the online decision)
- SE(3) error convention: `lab/se3.py` (translation metres, rotation vector radians)

## Mechanism

Toledo proposal PROP-NATIVE-04, converged on by the 2026-09-09 ultracode team meeting reading this
project through the Toledo lens and the readout_universe+readout_genesis lens after run6's
refutation. Three targeted fixes composed on top of runs 5-6's diagnosed defects:

1. **Magnitude**: `eps_{k,j} := min(q_k/lambda_j, CEILING)` — `q_k` is PROP-CONF-03's per-checkpoint
   Bonferroni conformal quantile (`cqts.safety.safe_multi_checkpoint_quantiles`, reused unchanged),
   calibrated offline on CALIBRATION only, replacing run5/6's fixed TRAIN-population CAP.
2. **Signal**: `Gamma_k := lambda_j * eps_{k,j}^2` — the Keystone quadratic-form energy, a real
   scalar rather than run6's non-toggling boolean.
3. **Accumulator**: `m_k := rho*m_{k-1} + iota_k*Gamma_k`, a decayed real-valued running sum
   (`rho=0.5`) instead of run6's hard reset-to-zero streak.

ACT fires at the first predeclared checkpoint (`k in {4,8,12,15}`) where BOTH `m_k >= theta` AND
the checkpoint's own PROP-CONF-03 certificate condition holds (disclosed reading of the registered
`q_{k_m} <= tau_i`, a units mismatch as literally typeset — see `lab/native_conformal_energy.py`
module docstring and RATIONALE.md for the full account). No ground truth is used anywhere in the
online decision.

## Frozen protocol

- TRAIN / CALIBRATION / FINAL TEST episodes: 40 / 40 / 40 (identical to runs 1-6)
- Split rule: identical to runs 1-6 (60 frames drawn seed 20260909, permuted seed 20260909, cut 20/20/20)
- Predeclared checkpoints: `{4, 8, 12, 15}` (reused byte-identical from run3), `alpha=0.1`
- `CEILING` ("weld/M.40.v1 ceiling"): `0.03274911721483188` (reused byte-identical as run5/6's own CAP)
- `rho = 0.5` (predeclared, disclosed simplest non-extreme decay rate, not tuned on any outcome)
- `theta` (per task, all three identical on TRAIN): `0.0007464025551635286` (median TRAIN `m_k` at
  the final checkpoint, `k=15`)
- non-inferiority margin: 0.05 (reused unchanged)
- Predeclaration commit `a76a8b9` predates `lab/run_native_conformal_energy.py`'s only invocation
  against `test.jsonl` for run7 (F4 discipline maintained)

## Honest prediction stated BEFORE opening test.jsonl (see RATIONALE.md)

TRAIN+CALIBRATION-only diagnostic (`lab/derive_native_conformal_energy_from_train.py`, never
opened `test.jsonl`) found: (1) `eps_{k,j}` saturates at `CEILING` on 100% of the 160 TRAIN
stage-checkpoint pairs for all three tasks — `q_k/lambda_j` is uniformly LARGER than the reused
ceiling, so the conformal-quantile term never actually influences the magnitude, and `eps_{k,j}`
collapses to being numerically identical to run5/6's plain CAP; (2) gate (b) (the checkpoint's own
PROP-CONF-03 certificate condition) is TRUE on 0/40 TRAIN episodes at all 4 checkpoints for all 3
tasks — reproducing run3's already-known `certificate_rate=0.0` finding. This predicted, before
test.jsonl was read, that run7 would fail closed to ~100% HOLD (ACT rate near 0%) — the OPPOSITE
failure mode from runs 5-6. The test-time result below confirms that prediction exactly.

## Primary results

| Task | ACT rate | HOLD rate | ACT-endpoint completion | estimator completion | unsafe ACT | non-inferiority |
|---|---:|---:|---:|---:|---:|---|
| top_suction | 0% | 100% | 0.0% (no ACT episodes) | 32.5% | 0/40 (0.0%) | FAILS (LCB -49.1%) |
| label_alignment | 0% | 100% | 0.0% (no ACT episodes) | 27.5% | 0/40 (0.0%) | FAILS (LCB -43.9%) |
| keyed_insertion | 0% | 100% | 0.0% (no ACT episodes) | 10.0% | 0/40 (0.0%) | FAILS (LCB -23.7%) |

ACT never fires on any of the 40 test episodes, for any of the 3 tasks, at any of the 4 predeclared
checkpoints. Mean endpoint = 15 (the terminal ICP stage) for all three tasks, i.e. every episode
runs the full refinement budget and then HOLDs.

## Mandatory failure accounting

- **Number of unsafe ACTs total: 0/40 for all three tasks (0%).** This is the OPPOSITE failure mode
  from runs 5-6 (which found 82.5-95.0% unsafe ACT): run7 never commits to a wrong answer, because
  it never commits to anything.
- **Number of HOLD episodes: 40/40 (100%) for all three tasks.** ACT never fires within the
  15-iteration budget — the over-conservatism failure mode explicitly flagged as a risk to check by
  the run-7 implementation plan and by the Toledo entry's own honest_caveats ("Open risk 1
  (over-correction) ... HOLD too long or never ACT at all").
- **Stage-at-ACT distribution: empty — ACT never fires**, so there is no distribution to report
  beyond "never, at any of the 4 checkpoints, for any episode."
- `degenerate_h_k_stage_rate`: 0.0 for all three tasks — `H_k` was never numerically singular; this
  is the ordinary (non-degenerate) branch of the rule, same as runs 5-6.
- Gate diagnostics (checkpoint-level, 160 checkpoint readings = 40 episodes x 4 checkpoints per
  task): `m_k >= theta` alone held on 72/160 readings (45%) for every task; the certificate
  condition (gate b) held on **0/160 readings, for every task**; both gates together: **0/160**.
  ACT's binding constraint is gate (b) alone, at every single checkpoint, for every episode.

## Comparison against runs 5-6

| Metric | run5 (theta=1,k=0) | run6 (theta=4,k=3,fixed) | run7 (PROP-NATIVE-04) |
|---|---:|---:|---:|
| ACT rate | 100% | 100% | **0%** |
| unsafe ACT rate (mean of 3 tasks) | 95.0% | 82.5% | **0%** |
| HOLD rate | 0% | 0% | **100%** |
| non-inferiority | FAILS (unsafe-ACT direction) | FAILS (unsafe-ACT direction) | FAILS (over-conservative direction) |

Runs 5-6 failed by ACTing too eagerly and being wrong. Run7 fails by never ACTing at all — it trades
one honest failure mode for its polar opposite, exactly as the Toledo entry's own "Open risk 1"
anticipated as a real possibility to check.

## Root-cause diagnosis (still fails non-inferiority, opposite direction)

**PROP-NATIVE-04's two intended fixes did not compose the way the design hoped, for two distinct,
now precisely located reasons, both predicted from TRAIN+CALIBRATION data before test.jsonl was
opened:**

1. **The conformal-quantile magnitude fix never activates.** `q_k` (order ~2.1-2.2, a dimensionless
   log-nonconformity quantile) divided by `lambda_j` (typically well below 1 on this backend) is
   essentially always larger than the reused `CEILING` (0.0327), so `min(q_k/lambda_j, CEILING)`
   picks `CEILING` at every stage-checkpoint pair observed — on both TRAIN and TEST. The magnitude
   term `eps_{k,j}` is therefore numerically IDENTICAL to run5/6's plain CAP throughout; the
   conformal quantile contributes nothing to the perturbation-magnitude half of this construction on
   this backend/dataset. This is a direct consequence of reusing run5/6's own CEILING value
   byte-identical (per the Toledo entry's own "Open risk 4" instruction) — the ceiling itself was
   already diagnosed by run5 as too small relative to the real coarse-detector error scale, and
   capping the new, larger conformal-quantile term at that same small ceiling reproduces exactly the
   magnitude run5 already found undersized. The Keystone-energy/decay-accumulator half of the
   construction (fixes 2 and 3) therefore never gets to operate on a genuinely new magnitude signal;
   it operates on run5/6's already-refuted one.
2. **Gate (b) — tying the decision back to the task's own admissible region — never holds, because
   it inherits PROP-CONF-03's own already-known-vacuous calibration.** Run3 already found
   `certificate_rate=0.0` (100% HOLD) for this exact backend/model/checkpoint set/alpha: the
   whole-trajectory-avoiding, per-checkpoint Bonferroni-corrected conformal envelope is still too
   wide to ever fit inside the declared task tolerances on this dataset. PROP-NATIVE-04 reuses that
   SAME calibrated envelope as its second, conjunctive ACT gate, so it inherits that 0%-certificate
   finding unchanged: gate (b) is false at every single checkpoint reading in both TRAIN and TEST,
   making it the sole binding constraint on ACT regardless of what the native/Keystone/decay half of
   the construction computes (`m_k >= theta` alone was true on 45% of checkpoint readings — the
   Keystone/decay half is not vacuous by itself, but the conjunction with gate (b) is).

**Both root causes were disclosed in the predeclaration commit (`a76a8b9`, RATIONALE.md and
`native_conformal_energy_threshold_derivation.json`) before `test.jsonl` was ever opened for run7** —
this is not a post-hoc discovery.

## What this does and does not establish

**Does NOT establish:** that PROP-NATIVE-04 is a viable fix for runs 5-6's diagnosed problems — it is
refuted, in the specific, disclosed implementation tested here (checkpoint-indexed accumulator,
gate (b) read as PROP-CONF-03's own certificate condition, CEILING reused byte-identical from
run5/6). It does NOT establish that combining a calibrated conformal quantile with a native/local
check can never work in principle — only that THIS specific composition, on THIS backend/dataset,
inherits both of its two ingredient constructions' own already-diagnosed weaknesses (run5/6's
undersized ceiling, still binding because it was reused unchanged rather than raised; PROP-CONF-03's
own too-wide whole-checkpoint envelope, still never fitting inside task tolerance) rather than
canceling them out.

**Does establish:** (1) PROP-NATIVE-04, implemented per the disclosed reading above, is REFUTED —
ACT rate 0% (100% HOLD) on all three declared tasks, failing non-inferiority in the over-conservative
direction (LCB -23.7% to -49.1%). (2) The predicted failure was stated, with the correct mechanism,
before test.jsonl was opened — both diagnosed causes (ceiling saturation, gate-b vacuity) were found
from TRAIN+CALIBRATION data alone. (3) A future construction along this line needs either (a) a
genuinely raised ceiling (not run5/6's reused, already-too-small value) so the conformal quantile's
larger magnitude can actually take effect, or (b) a materially different certificate condition than
PROP-CONF-03's own per-checkpoint envelope for gate (b), since that envelope is now independently
confirmed vacuous on this backend/dataset across two entirely different constructions (run3's
whole-checkpoint certificate, and this run's conjunctive gate) — a within-project stability finding,
not a coincidence of one run's tuning.

## Novelty/positioning (unchanged guardrails)

This result does not "solve 6D pose estimation" and is not claimed as first anything. Consistent
with `ops/HANDOFF_2026-09-09_external_dataset.md`'s positioning constraint: "We are not trying to
eliminate perception error entirely. We ask whether the error that remains still affects what the
robot is about to do." This run's honest answer is: the construction tested here answers that
question by refusing to answer it at all, on every held-out trial, for every declared task.

## Claim form

On the in-repo numpy+scipy ICP backend / BOP LM-O real depth data / CPU, PROP-NATIVE-04's
conformal-scaled Keystone-energy decay accumulator (`rho=0.5`, checkpoints `{4,8,12,15}`,
`CEILING=0.0327`, `theta` from TRAIN) never licensed ACT on any of 40 held-out trials, across all
three declared tasks — a genuinely different, disclosed, and predicted-before-test failure mode
(100% HOLD) from runs 5-6's unsafe-ACT failure (82.5-95.0% unsafe), attributable to two independently
diagnosed causes: the reused ceiling still capping the perturbation magnitude below the conformal
quantile's actual scale, and the checkpoint certificate gate inheriting PROP-CONF-03's own
already-known-vacuous calibration. Reported as a refuted, safety-relevant negative result — zero
unsafe ACTs is not a positive certificate-works claim when it is achieved by never acting at all.

## Physical robot extension

Not applicable — no physical manipulation was executed. Pose-ground-truth replay only.
