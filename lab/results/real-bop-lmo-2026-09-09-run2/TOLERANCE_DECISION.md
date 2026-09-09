# Tolerance decision — run2 (2026-09-09)

This file is committed in the SAME commit as `config.json` and
`split_manifest.json`, and that commit predates any invocation of
`lab/run_real_system.py` against `test.jsonl` for this run. Its purpose is to
close the F4 process gap disclosed in the v0.7.0 release notes: last time the
frozen config was committed alongside `lab_results.json` in one commit, not
before it.

## The two options considered

1. **Reuse** the exact same task tolerances as the first run
   (`lab/results/real-bop-lmo-2026-09-09/config.json`), byte-identical, to
   test whether the 0% certificate rate reproduces. This is a legitimate
   reproducibility check on its own.
2. **Re-derive** tolerances from the real ape/driller objects' actual
   achievable ICP accuracy within the same 15-iteration budget, using ONLY
   TRAIN-split data, never looking at CALIBRATION or FINAL TEST error
   statistics while deriving them.

## Decision: re-derive (option 2)

Reused unchanged, the first run's own tolerances (5mm/5mm/8mm translation,
~5deg rotation on most axes) sit below the median TRAIN-split achievable
error (see below): the ICP backend, on this real sensor data, in this
15-iteration budget, typically does **not** land inside those tolerances even
at its own best convergence stage. Re-running FINAL TEST with identical
tolerances would deterministically reproduce the same 0% certificate result
for a reason already established by run 1's own diagnosis (RESULT.md,
"Explicit limits": *"the observed 0% certificate rate may reflect tolerances
that are simply too tight for this backend's real achievable accuracy...
rather than a fundamental property of coverage-qualified stopping"*) — running
it again would not test a new hypothesis, it would just re-confirm a
diagnosis already made without spending a second real-data cycle on it.

Re-deriving lets this run separate two previously confounded questions:
(a) is the observable-feature/conformal-envelope machinery itself unable to
produce a useful (non-vacuous, discriminative) completion certificate on real
ICP data, versus (b) were the numerical-fixture tolerances simply the wrong
absolute numbers for this real backend/object/budget combination. Run 1 could
not distinguish these. This run is designed to distinguish them.

## Derivation method (reproducible from TRAIN alone)

Full method and code: `lab/derive_tolerances_from_train.py`. Summary:

1. For each of the 40 TRAIN episodes (`lab/data/bop_lmo_episodes/train.jsonl`,
   unchanged from run 1 — same frames, same objects, same ICP runs), take the
   stage index returned by `lab.run_real_system.estimator_index` — the
   backend's own convergence stage within the 15-iteration budget, or the
   terminal stage if it never converges. This is the accuracy the backend
   actually delivers under its own stopping rule, not a best-case stage pick.
2. Read that stage's oracle `abs_pose_error_6d` (TRAIN oracle access is within
   the lab protocol's own design — `lab/README.md` section 3: "TRAIN fits the
   observable error-shape model" — this is not calibration/test leakage).
3. Take the 75th percentile of each of the 6 error components across the 40
   TRAIN episodes. p75 (not p50/p90) was fixed as the summary statistic
   **before** running the script: "the backend should meet this tolerance on
   a majority-plus of typical trials," a middle choice between "typical
   trial" (p50 — too close to a coin flip to state as a working tolerance)
   and "worst realistic trial" (p90 — arbitrarily loose).
4. Round the p75 value UP (ceiling) to a coarse, human-legible grid: 1 mm for
   translation, 1 degree for rotation. This rounding rule was fixed before
   any number was computed, so the exact tolerance values below cannot be
   read as reverse-engineered from an outcome.

This computation touches TRAIN only. It was run and its numbers frozen into
`config.json` before `lab/run_real_system.py` was invoked on
`test.jsonl` for run2 — see the commit that carries this file for the exact
commit hash/timestamp that predates that evaluator invocation.

## Resulting per-component tolerance (from `tolerance_derivation.json`)

| component | run-1 tolerance | TRAIN p75 (raw) | run-2 tolerance (rounded up) |
|---|---:|---:|---:|
| tx | 5 mm | 13.48 mm | 14 mm |
| ty | 5 mm | 12.48 mm | 13 mm |
| tz | 8 mm | 12.19 mm | 13 mm |
| rx | 5.0 deg | 6.97 deg | 7 deg |
| ry | 5.0 deg | 7.33 deg | 8 deg |
| rz | 3-4 deg (task-dependent) | 8.32 deg | 9 deg |

Task masks and reader `kind` (box/l1) are unchanged from `lab/config.example.json`
and from run 1's `config.json` — only the numeric tolerance for the
components each task's mask actually uses was replaced with the value above.
Components a task's mask does not use keep the original `1e9` sentinel.

## What this decision does NOT do

- It does not touch CALIBRATION or FINAL TEST data in any way.
- It does not change the split, the frames, the objects, the ICP backend, the
  observable feature vector, the estimator-side stopping rule, or the
  refinement budget — all of those are reused unchanged from run 1, so the
  tolerance choice is the only varied factor between run 1 and run 2.
- It is not tuned to produce a nonzero certificate rate — the derivation
  script was run exactly once, its output was not adjusted after seeing it,
  and it was computed and committed before FINAL TEST was opened.
