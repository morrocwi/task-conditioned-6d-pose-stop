# Predeclaration — real-bop-lmo-2026-09-09-run5 (PROP-NATIVE-01/02, H3)

Fifth real-data cycle. Founder authorization
(`ops/HANDOFF_2026-09-09_external_dataset.md`, "Fifth cycle authorized, 2026-09-09: H3 (native
retained-sensitivity model)"): "เราสร้างโมเดลมา แก้ปัญหาให้ certificate ทำงาน" -- build the model, solve
the problem so the certificate works, using the native no-T* model per Toledo proposals
PROP-NATIVE-01/02 (`~/ANSE.ASIA/toledo/registry/proposals/native_retained_sensitivity.json`).

## What is genuinely different from runs 1-4

Runs 1-4 all shared the same decision skeleton: fit or derive an error-scale predictor on TRAIN,
calibrate a conformal quantile on CALIBRATION, and compare a calibrated completion-error bound
against a task tolerance. Every one of the four found `certificate_rate = 0.0` (100% HOLD) --
either the tolerances (run1/run2) or the calibration construction (run2-4) were implicated, never
the underlying premise that the whole mechanism needs a completion set calibrated against an
(offline) notion of true pose error at all.

This run replaces that premise, per PROP-NATIVE-01's own framing: the classical pose error
`e_k := Log(That_k^-1 T*)` is a distance to an unobserved `T*`, a non-readout under this
workspace's own information-discrete-math discipline. `lab/native_sensitivity.py` implements
PROP-NATIVE-02 instead: at stage k, take the real ICP normal-equations Hessian `H_k = J^T J`
(already exposed for run4 by `lab/bop_icp_backend.py:normal_equations_H`, unchanged), find its
single smallest eigenvalue/eigenvector `(lambda_min, v_min)` (reusing run4's own eigen-extraction
routine family, `lab/decay_predictor.py:eigh_H`, added additively -- `kappa_rho_from_H` only ever
returned the two extreme eigenvalues as scalars, never the eigenvectors this cycle needs), perturb
`T_hat_k` by `+-m*v_min` with `m = min(C/lambda_min, CAP)`, and ACT iff
`cqts.safety.fail_closed_task_pass` (reused unchanged) gives the SAME verdict for the unperturbed
pose and both perturbed candidates. **No ground truth is used anywhere in this decision.** There is
no fitted model, no calibrated completion set, no conformal quantile -- this is a structurally
different mechanism, not a variant of C6-C10.

## Held fixed, reused byte-identical from runs 2-4

- Dataset, frames, seeds, split (60 frames drawn seed 20260909, permuted seed 20260909, cut
  20/20/20 TRAIN/CALIBRATION/FINAL TEST) -- confirmed identical `split_manifest.json` between
  `lab/data/bop_lmo_episodes_decay/` (run4) and the newly generated
  `lab/data/bop_lmo_episodes_native/` (run5; the only difference is a `"native"` field per stage
  replacing run4's `"decay"` field: `H_k` flattened row-major, plus `T_hat_k=(R_k,t_k)`).
- ICP backend (`lab/bop_icp_backend.py`) -- unchanged since run4; the only backend-adjacent
  addition is `lab/adapter_bop.py:export_episode_native`/`native_sensitivity_stage_fields`, a
  purely additive export function.
- Task tolerances, `error_floor`, inference config (margin/confidence/minimum_n): reused
  byte-identical from run2/run3/run4's `config.json`.
- `alpha`/coverage fields do not apply to this mechanism (no calibrated quantile exists) and are
  dropped from this run's `config.json`; everything else in the frozen-config shape is unchanged.

## Predeclared eigenvalue-threshold rule

**Exactly the single smallest eigenvalue of `H_k` and its eigenvector** -- the simpler of the two
options the handoff explicitly offered ("the single smallest, or all below some declared fraction
of lambda_max"). Chosen to avoid any post-hoc argument about where a fraction threshold "should"
sit; a single, least-discretionary direction per stage.

Both signs of the eigenvector are tested (`+v_min` and `-v_min`), since `eigh`'s sign convention is
arbitrary and the true "data-justified-uncertain" set along that axis is two-sided, not one-sided.

## Predeclared perturbation-magnitude rule and cap, and how they were chosen

`m = min(C / lambda_min, CAP)`, `C := CAP * lambda_ref`. Both `lambda_ref` and `CAP` are derived
from **TRAIN only** by `lab/derive_native_threshold_from_train.py` (see
`native_threshold_derivation.json` in this directory for the exact numbers, committed alongside
this file):

- `lambda_ref = 0.33927484832749355` = median(lambda_min(H_k)) over all 640 TRAIN stage readings.
- `CAP = 0.03274911721483188` = the 75th percentile of the Euclidean norm of the online-observable
  local-dispersion proxy (`lab/bop_icp_backend.py:local_proxy`, the SAME 6-component proxy already
  carried in `observable_features` and already used by runs 1-4's own fitted model / run4's decay
  predictor) over all 640 TRAIN stage readings.

### An earlier candidate CAP was discarded, honestly, before opening test

The first candidate tried was a hand-picked round number, `CAP=0.005` (5mm / ~0.29deg), chosen
because it sat comfortably below every task tolerance component (13-14mm translation, ~7-9deg
rotation on the used axes) -- reasoning, at the time, that a "small, physically modest" bound was
the conservative choice. A CALIBRATION-only diagnostic (`lab/data/bop_lmo_episodes_native/
calibration.jsonl`, never `test.jsonl`) run during code development immediately showed this was a
**definitional flaw, not a genuine finding**: because CAP was smaller than every masked task
tolerance component, EVERY perturbed candidate at EVERY stage trivially passed the task reader
regardless of `lambda_min`, so the mechanism ACTed at stage k=0 on effectively 100% of calibration
episodes independent of any real geometric signal -- a vacuous test, not evidence about H3. Sweeping
CAP from 0.005 up through 0.1 on calibration reproduced the identical vacuous act/unsafe numbers at
every value in that range, confirming this was not a narrow edge case.

This is disclosed here as the discarded step it was -- not silently dropped, and not "tuned toward
an outcome": the reasoning for discarding it is definitional (a cap smaller than every tolerance
makes the invariance test unable to ever fail, by construction, independent of what value ends up
observed on any split), not a preference for a nicer number on any split's results. The replacement
CAP (p75 of a real, already-existing observable per-stage statistic, computed from TRAIN alone) was
adopted because it ties the bound to a physically meaningful quantity instead of an arbitrary round
number, and was frozen without ever reading calibration OUTCOME statistics (act rate, correctness,
etc.) at that value, let alone test.jsonl, before this commit.

### Fail-closed handling of a degenerate (singular) `H_k`

If `lambda_min` is non-finite or `<= 0` (H_k numerically singular -- a real, unremarkable situation
early in ICP or with few inliers, exactly as `lab/decay_predictor.py:kappa_rho_from_H` already
treats it as fail-closed to `kappa=+inf`), `m` is simply set to `CAP` -- bounded, never unbounded,
never silently treated as zero-uncertainty. `lab/run_native_sensitivity.py` additionally reports the
`degenerate_h_k_stage_rate` per task so this never happens silently.

## Known, disclosed limitations of this construction

- SE(3) is still a continuum-manifold representation for `T_hat_k`
  (`theory/CONTINUUM_AUDIT_20260909.md`, item 1). This cycle removes `T*` from the online decision
  (H3's whole point) but does NOT fix that deeper continuum injection -- a separate, unattempted
  design pass.
- A single `H_k` eigenvector mixes 3 rotation (radians) and 3 translation (metres) components,
  perturbed by ONE shared scalar magnitude without separate nondimensionalization -- the same
  honesty gap already disclosed for `kappa` in `docs/NAVIER_STOKES_THROUGH_OUR_LENS.md`'s
  kappa-scaling honesty note. This is carried over deliberately, not silently assumed away.
- No proven floor exists for "how small a `lambda_min` is too small": `q_formal/M.07`'s
  diameter-based floor was already shown in run4 not to apply to `H_k`'s structure at all (no
  vertex/edge structure), and remains unmechanized regardless (its own Coq attempt did not close).
  `CAP` here is a declared/relative, TRAIN-derived bound -- not a bound resting on any proven floor.
- PROP-NATIVE-02's own `honest_caveats` flag the central empirical assumption this run tests:
  "the relationship between 'small eigenvalue direction' and 'actual pose error along that
  direction' is itself an assumption ... must be checked empirically." This run IS that check --
  see `RESULT.md` for the outcome.

## No parameters tuned by peeking at final-test outcomes

`lambda_ref` and `CAP` are closed-form statistics of TRAIN data alone, computed once by
`lab/derive_native_threshold_from_train.py` and never revisited after this commit. The one
CALIBRATION-only exploration described above changed the CAP-selection METHOD (from "a hand-picked
round number below tolerance" to "p75 of an existing observable statistic") for a definitional
reason stated before any calibration OUTCOME number was inspected at the replacement value; no
parameter was subsequently adjusted after seeing the replacement's calibration-split numbers.
`test.jsonl` was not read, in any form, before this commit.

## Commit discipline (F4, matching runs 2-4)

This file, `config.json`, `split_manifest.json`, and `native_threshold_derivation.json` are
committed in their OWN commit, **before** `lab/run_native_sensitivity.py` is invoked against
`test.jsonl` for this run. `lab/data/bop_lmo_episodes_native/{train,calibration,test}.jsonl`
(deterministic, seed-driven, regenerated by `lab/generate_bop_episodes.py --variant native`) is
committed alongside, since it is itself part of the frozen input, not a final-test output.
