# Predeclaration rationale — real-bop-lmo-2026-09-09-run3

Third real-data cycle. Tests the PROP-CONF-03 Bonferroni-corrected multi-checkpoint conformal
band, designed to address the cause diagnosed in `~/ANSE.ASIA/glosa/projects/GLS-2026-005_pose-stop-conformal-diagnosis/DIAGNOSIS_HYPOTHESIS.md`
for runs 1-2's `certificate_rate = 0.0` on all three tasks. This file is committed together with
`config.json`, `split_manifest.json` (copied unchanged from run2), and `backend_and_hardware.md`,
in one commit, **before** `lab/run_real_system.py --mode bonferroni_multicheckpoint` is ever
invoked against `test.jsonl` for this run — following the F4 process fix already applied in run2.

## What is fixed by design (not chosen by looking at this run's data)

- **K' = 4 checkpoints.** `cqts.safety.bonferroni_feasible_max_checkpoints(40, 0.1) == 4`: with
  n=40 calibration episodes and alpha=0.1, K'=4 is the largest feasible checkpoint count
  (`ceil(41*(1-0.1/4)) = 40 <= 40`); K'=5 is already infeasible (`ceil(41*(1-0.1/5)) = 41 > 40`,
  forces q=+infinity at every checkpoint). This arithmetic was computed and recorded in the
  GLS-2026-005 diagnosis revision (2026-09-09) BEFORE this run's code existed — it is a
  mechanical feasibility bound, not a tuned choice.
- **Checkpoint stages = {4, 8, 12, 15}.** Chosen to span the 15-iteration refinement budget
  roughly evenly (early/mid/late/terminal), matching the diagnosis's own example
  (`stages {4, 8, 12, 15} or similar`). No alternative spacing was scored against any
  calibration/test statistic before this choice.
- **Nonconformity score = coordinate-max only, per checkpoint, never a max over stages.** This is
  the literal PROP-CONF-03 construction (Toledo registry
  `~/ANSE.ASIA/toledo/registry/proposals/conformal_stopping_family.json`), implemented in
  `lab/multicheckpoint.py::nonconformity_at_checkpoint`, calibrated via
  `cqts.safety.safe_multi_checkpoint_quantiles` which is a thin per-checkpoint dispatcher over the
  UNCHANGED `cqts.safety.safe_split_conformal_quantile` — the conformal quantile order-statistic
  logic itself is reused, not reimplemented.

## Data, split, backend, tolerances: reused unchanged from run 2

- **Identical BOP-LMO data/split/seeds** as runs 1-2 (`lab/data/bop_lmo_episodes/{train,
  calibration,test}.jsonl`, not regenerated). This isolates the certificate-construction method as
  the ONLY varied factor across all three real-data cycles.
- **Identical ICP backend** (`lab/bop_icp_backend.py`, `lab/adapter_bop.py`), unchanged commit.
- **Task tolerances: reused byte-identical from run 2** (TRAIN-derived: 75th percentile of the
  estimator-convergence-stage absolute pose error across the 40 TRAIN episodes, rounded up to
  1mm/1deg — `lab/results/real-bop-lmo-2026-09-09-run2/TOLERANCE_DECISION.md`), **not** run 1's
  original numerical-fixture tolerances. Rationale for this choice: this run tests the
  AGGREGATION fix (does per-checkpoint Bonferroni calibration shrink the calibrated envelope
  enough to certify), not the tolerance question again — run 2 already validated that the
  TRAIN-derived tolerances capture real achievable ICP accuracy for these objects (they raised the
  estimator's own completion rate substantially, 2.5%->32.5% / 2.5%->27.5% / 0%->10%), and reusing
  the numerical-fixture tolerances again would reintroduce a confound (too-tight tolerance vs.
  too-wide envelope) that run 2 already ruled out as the dominant driver. Reusing run 2's
  tolerances keeps this run a clean single-factor test of the aggregation-scheme change.
- `alpha = 0.1`, `error_floor`, `inference` block (`margin=0.05`, `confidence_level=0.95`,
  `minimum_test_episodes=30`, `independent_sampling_unit="episode"`), `timing_mode =
  trajectory_prefix_estimate`: all unchanged from run 2, for the same reason (isolate one factor).

## Falsifiable prediction this run tests (from the diagnosis, stated before running)

> If the whole-trajectory max-aggregation (over 15 stages, C7/PROP-CONF-02) is the dominant driver
> of the ~7.4x envelope inflation seen in runs 1-2 (joint `q ~ 2.0008`), then the per-checkpoint
> Bonferroni-corrected quantiles `q_{alpha/K'}` at each of the 4 predeclared checkpoints should be
> smaller than the joint `q ~ 2.0008`, and the certificate rate should rise above 0% on the SAME
> test episodes without touching task tolerances again.

Refutation of this prediction (per-checkpoint q's not smaller, or certificate rate still 0%) is an
equally reportable, equally informative outcome — see `RESULT.md`.

## Disclosed process note (honesty, not concealment)

During code development, `lab/run_real_system.py --mode bonferroni_multicheckpoint` was invoked
once against the REAL `test.jsonl` (with `checkpoints=[4,8,12,15]` and run2's tolerances already
fixed as above) as a smoke test to confirm the new code path executes without error end-to-end.
This incidentally surfaced the real per-checkpoint q values and task outcomes for this exact
configuration BEFORE this predeclaration commit — a process gap similar in kind to run 1's F4
(though narrower: no free parameter documented here was subsequently changed in response to that
smoke test; K'=4, checkpoints={4,8,12,15}, and the reused run-2 tolerances were already fixed, by
the diagnosis document and this rationale's own arguments above, before that smoke-test
invocation). Stated plainly rather than hidden: a fully clean process would have smoke-tested the
new code only against synthetic fixture episodes (as `tests/test_multicheckpoint.py` does), never
against the real held-out test split, before this freeze commit. This is disclosed as a real
methodological imperfection in this run's own process, not retroactively excused.
