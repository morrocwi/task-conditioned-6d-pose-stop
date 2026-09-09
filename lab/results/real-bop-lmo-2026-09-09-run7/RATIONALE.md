# Predeclaration — real-bop-lmo-2026-09-09-run7 (PROP-NATIVE-04)

Seventh real-data cycle. Founder authorization (2026-09-09, verbatim): "ultracode ประชุมทีมด้วยความรู้เดิม
ที่มีและอ่านงานด้วยเลนส์ toledo ของเราและ readout universe+genesis แล้วหาคำตอบให้เจอ ขยายการทดลองเพิ่มตามข้อค้นพบและ
สมมุติฐานใหม่" — convene an ultracode team meeting using existing knowledge, read the work through the
Toledo lens and the readout_universe+readout_genesis lens, find the answer, and expand the experiment
following the findings and new hypothesis. The team meeting converged on Toledo proposal PROP-NATIVE-04
(`~/ANSE.ASIA/toledo/registry/proposals/native_retained_sensitivity.json`), which this run tests.

## What is genuinely different from runs 5-6

PROP-NATIVE-04 composes three targeted fixes, one per diagnosed defect:

1. **Magnitude** (fixes run 5): the perturbation magnitude's numerator is now `q_k`, PROP-CONF-03's
   already-calibrated per-checkpoint Bonferroni conformal quantile, instead of run 5/6's fixed
   TRAIN-population CAP alone. `eps_{k,j} := min(q_k/lambda_j, CEILING)`.
2. **Signal** (fixes run 6): the per-stage quantity fed to the accumulator is `Gamma_k :=
   lambda_j*eps_{k,j}^2` (the Keystone quadratic-form energy of the actual perturbation), a real
   scalar, not a 0/1 flag.
3. **Accumulator** (fixes run 6): `m_k := rho*m_{k-1} + iota_k*Gamma_k` is a decayed real-valued
   running sum, not a hard reset-to-zero streak.

See `lab/native_conformal_energy.py`'s module docstring for two disclosed interpretation choices
this implementation had to make explicit, neither of which was left ambiguous or silently resolved:

- **Checkpoint-indexed, not per-ICP-iteration**: `q_k` only exists at the K'=4 predeclared
  Bonferroni checkpoints (reused byte-identical from run3: `{4,8,12,15}`, `alpha=0.1`), so `m_k`
  accumulates once per checkpoint in that order, not once per ICP iteration.
- **"q_{k_m} <= tau_i" read as PROP-CONF-03's own certificate condition**: the registered LaTeX's
  literal scalar comparison mixes a dimensionless log-nonconformity quantile against a physical
  tolerance — a units mismatch, not a sensible inequality as typeset. This is read instead as the
  already-implemented, physically well-typed condition `lab/multicheckpoint.py:
  first_certificate_checkpoint` already evaluates: the calibrated completion-error bound built from
  `q_k` lies inside the task-admissible region.

## Held fixed, reused byte-identical from runs 2-6

- Dataset, frames, seeds, split (60 frames drawn seed 20260909, permuted seed 20260909, cut
  20/20/20 TRAIN/CALIBRATION/FINAL TEST) — `split_manifest.json` copied unchanged from run6.
- ICP backend (`lab/bop_icp_backend.py`), task reader kind/mask/tolerances, `error_floor`,
  inference config (margin/confidence/minimum_n).
- `lambda_ref`/native-sensitivity eigen-extraction machinery (`lab/native_sensitivity.py`, reused
  unmodified) — used here ONLY to supply `lambda_j` and the CEILING (see below), not as its own
  independent stopping rule.
- PROP-CONF-03's Bonferroni checkpoint machinery (`cqts/safety.py`, `lab/multicheckpoint.py`, both
  already implemented and reused unmodified — the plan's "check first before writing new code"
  instruction found this already existed, contrary to the Toledo entry's own "not yet implemented"
  note on PROP-CONF-03 itself, which refers to it never having been *empirically tested end-to-end
  as PROP-NATIVE-04's own magnitude source before this run*, not to missing code).

## New free parameters, predeclared from TRAIN alone (honest_caveats Open risk 5)

- **CEILING ("weld/M.40.v1 ceiling")** = `0.03274911721483188` — reused byte-identical as run5/6's
  own TRAIN-derived CAP. Per the Toledo entry's own honest_caveats ("Open risk 4 ... the same
  UNPROVEN bridge PROP-NATIVE-02 already carried"), this proposal does not mint a new numeric
  ceiling; it reuses the existing registered one.
- **rho = 0.5**: predeclared before any data was read. Disclosed reasoning: the simplest non-extreme
  decay rate in (0,1) — avoids both `rho=0` (no memory, reduces to a single-instant check like run5)
  and `rho=1` (infinite memory, reduces to run6's non-decaying hard sum). Not tuned by looking at
  any TRAIN, calibration, or test outcome.
- **theta** (per task, see `native_conformal_energy_threshold_derivation.json`): the median, over
  the 40 TRAIN episodes, of the checkpoint-indexed accumulator `m_k` evaluated at the final
  predeclared checkpoint (`k=15`) with `rho=0.5` fixed above. All three tasks yield the identical
  value `0.0007464025551635286` on TRAIN (the accumulator depends on `lambda_min(H_k)` and the
  reused CEILING, not on the task spec, whenever `iota_k` is True at every checkpoint — see below).

## TRAIN+CALIBRATION-only diagnostic, disclosed BEFORE test.jsonl was opened

`lab/derive_native_conformal_energy_from_train.py` computes, using TRAIN (for `H_k` eigenvalues and
model fitting) and CALIBRATION (for `q_k`, the standard split-conformal recipe already predeclared
by runs 1-3 — never test.jsonl), two structural findings:

1. **`eps_{k,j}` saturates at CEILING on 100% of the 160 TRAIN stage-checkpoint pairs (40 episodes
   x 4 checkpoints), for all three tasks.** `q_k` is order ~2.1-2.2 (a dimensionless log-space
   nonconformity quantile — see `q_by_checkpoint` in the derivation JSON) and `lambda_min(H_k)` is
   typically well below 1, so `q_k/lambda_min` is essentially always far LARGER than the reused
   CEILING (0.0327) — the opposite direction from run5's original diagnosis (there, the CAP itself
   was too small relative to real error; here, the new conformal numerator is even larger, so the
   `min()` still always picks the same small CEILING). **The conformal-quantile term therefore never
   actually influences the perturbation magnitude on this backend/dataset: `eps_{k,j}` collapses to
   being numerically IDENTICAL to run5/6's plain CAP at every stage-checkpoint pair observed.**
2. **Gate (b) (the checkpoint's own PROP-CONF-03 certificate condition) is TRUE on 0/40 TRAIN
   episodes, at all 4 checkpoints, for all 3 tasks** — reproducing run3's already-known
   `certificate_rate=0.0` finding on this exact backend/model/checkpoints/alpha, now checked
   directly on TRAIN rather than only recalled from run3's TEST result.

**Honest prediction stated before test.jsonl was opened:** because ACT requires BOTH `m_{k_m}>=theta`
AND gate (b), and gate (b) never holds on TRAIN at any predeclared checkpoint for any task, this
predicts ACT rate near 0% (100% HOLD) on FINAL TEST — the opposite failure mode from runs 5-6
(over-conservatism / never-ACT, not unsafe-early-ACT), consistent with the Toledo entry's own
disclosed "Open risk 1 (over-correction)" and "Open risk 2 (borrowed, untested calibration layer)".
This is a genuinely different failure mode from anything runs 1-6 produced by design, and it is
disclosed here, in the predeclaration commit, before any test-time number was read.

## Non-inferiority, task specs, timing methodology

Identical to runs 2-6: `cqts.safety.paired_binary_noninferiority`, margin 0.05, confidence 0.95,
minimum_test_episodes 30 (40 test episodes available), `trajectory_prefix_estimate` timing mode
(no latency-reduction PASS claimed).

Predeclaration commit predates `lab/run_native_conformal_energy.py`'s only invocation against
`test.jsonl` for run7 (F4 discipline maintained, as in every run since run2).
