# External Lab Result Report — real-bop-lmo-2026-09-09-run2

Filled per `lab/RESULT_TEMPLATE.md`. This run was executed **in-house**, in this
repository, not by an independent laboratory. Do not read it as external
validation. It is the repository's **second** real-sensor cycle, run to fix a
disclosed process gap (F4) from the first cycle
(`lab/results/real-bop-lmo-2026-09-09/`) and to test whether that first run's
0% certificate rate was an artifact of reusing numerical-fixture tolerances
that were simply too tight for this real backend, rather than a property of
the coverage-qualified stopping mechanism itself.

## Laboratory and system

- Institution: none (in-repo run, not an external lab)
- Laboratory: n/a
- Date: 2026-09-09
- Backend name: in-repo numpy+scipy point-to-point ICP/Kabsch (unchanged from run 1)
- Backend version / git commit: config/tolerance freeze committed at `d9ac505f40df81344ffe0e3737cb1b51d36bf13e`; backend code itself unchanged since `ed8735ad04c17efe1d1bc0a0cec4625071d2bdbb`
- Model weights identifier: none (no learned model)
- Camera / sensor: BOP LM-O test scene 000002's own recorded RGB-D sensor (not our sensor; archived dataset)
- GPU / CPU: AMD Ryzen 7 4800H (CPU only; GTX 1650 Ti present but unused)
- Operating system: Linux 7.0.0-30-generic x86_64
- Object set: BOP LM-O `obj_id` 1 (ape), 8 (driller) — same objects, same frames as run 1
- Task set: `top_suction`, `label_alignment`, `keyed_insertion` — same masks/reader kind as `lab/config.example.json`; tolerances RE-DERIVED from TRAIN, see `TOLERANCE_DECISION.md`
- Ground-truth source: BOP LM-O `scene_gt.json`
- SE(3) error convention: `lab/se3.py`, unchanged

## Frozen protocol

- TRAIN episodes: 40 (byte-identical to run 1)
- CALIBRATION episodes: 40 (byte-identical to run 1)
- FINAL TEST episodes: 40 (byte-identical to run 1)
- Split rule: unchanged from run 1 (see `split_manifest.json`, identical content) — 60 frames drawn from a 177-frame pool, permuted, cut 20/20/20
- `alpha`: 0.1 (unchanged)
- nominal whole-trajectory coverage: 0.9 (unchanged)
- non-inferiority margin: 0.05 (unchanged)
- estimator-side stopping rule: unchanged (two consecutive stages with |delta_t|<=0.5mm and |delta_r|<=0.1deg, else terminal stage)
- online feature vector definition: unchanged, see `config.json`
- task reader definition: same mask/kind as run 1; **tolerances changed** — re-derived from the 75th percentile of TRAIN-split estimator-convergence-stage absolute pose error, rounded up to 1mm/1deg (see `TOLERANCE_DECISION.md`, `tolerance_derivation.json`)
- deviations from `lab/README.md`: same two deviations as run 1 — `timing_mode=trajectory_prefix_estimate` (no latency claim); `inference.independent_sampling_unit=episode` while the dataset's true independent unit is the frame (20 independent test frames, not 40 test episodes)
- **Process change vs run 1 (fixes F4)**: `config.json`, `split_manifest.json`, `TOLERANCE_DECISION.md`, and `tolerance_derivation.json` were committed in their own commit (`d9ac505f40df81344ffe0e3737cb1b51d36bf13e`, 2026-09-09T12:53:35+07:00) **before** `lab/run_real_system.py` was invoked against `test.jsonl` for this run. No final-test result was inspected before that commit.

## Primary results

Held-out FINAL TEST whole-trajectory envelope coverage: **87.5%** (35/40), Wilson 95% CI [73.9%, 94.5%] — identical to run 1, because coverage depends on the calibrated envelope and calibration data, not on task tolerances; the calibration quantile is unchanged (`q ≈ 2.0008`, same as run 1).

| Task | test coverage | `k_C<k_E` | mean `k_C-k_E` [95% CI] | estimator completion | certificate completion | completion difference [95% CI] | HOLD | unsafe ACT | perception latency difference [95% CI] | non-inferiority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| top_suction | 87.5% | 0.0% | n/a (no certificate ever fired) | 32.5% | 0.0% | -32.5 pp | 100% | 0.0% | +0.177 ms [0.000, 0.530] (endpoint-k-based prefix, not a speed claim) | FAIL (LCB -49.1% < -5% margin) |
| label_alignment | 87.5% | 0.0% | n/a | 27.5% | 0.0% | -27.5 pp | 100% | 0.0% | +0.177 ms [0.000, 0.530] | FAIL (LCB -43.9% < -5% margin) |
| keyed_insertion | 87.5% | 0.0% | n/a | 10.0% | 0.0% | -10.0 pp | 100% | 0.0% | +0.177 ms [0.000, 0.530] | FAIL (LCB -23.7% < -5% margin) |

Loosening the task tolerances (2.6-2.8x on translation, ~1.4-1.8x on rotation,
derived from TRAIN-split achievable accuracy) raised the **estimator's own**
completion rate substantially (2.5%->32.5%, 2.5%->27.5%, 0%->10%), confirming
the derivation captured real achievable accuracy. It did **not** move the
certificate's completion rate at all (0% in both runs, all three tasks). The
non-inferiority lower confidence bound got *worse*, not better, because the
estimator's completion rate rose while the certificate's stayed at zero.

## Mandatory failure accounting

- Number of unsafe ACTs on covered episodes: 0 (all three tasks)
- Number of unsafe ACTs total: 0 (all three tasks) — certificate never fired, so no ACT occurred
- Number of HOLD episodes: 40/40 for all three tasks (100%)
- Cases where `k_C >= k_E`: 40/40 (certificate never occurred; `k_C` undefined/+infinity for every episode)
- Calibration/test coverage shortfall, if any: FINAL TEST coverage 87.5%, identical to run 1 (tolerances do not affect the coverage calibration, which is computed independently of task tolerances); Wilson 95% CI [73.9%, 94.5%] includes the 90% target
- Distribution-shift conditions tested: none; in-distribution only, same as run 1

## Claim form

On the in-repo numpy+scipy point-to-point ICP/Kabsch backend under BOP LM-O
scene 000002 (ape and driller, 40 held-out real-sensor test episodes, 15-iteration
budget), coverage-qualified stopping **did not** occur before estimator-side
stopping in any held-out trial for any of the three declared tasks
(`k_C < k_E` rate = 0%), **even after task tolerances were re-derived from
TRAIN-split achievable ICP accuracy** (loosened 2.6-2.8x on translation and
1.4-1.8x on rotation relative to run 1's reused numerical-fixture
tolerances). The calibrated completion envelope itself (quantile `q ≈ 2.0`,
unchanged from run 1, driven by the calibration data and observable-feature
model, not by task tolerances) remained too wide to ever satisfy either the
old or the new tolerances, so the certificate policy HOLD-ed on 100% of test
episodes for all three tasks in both runs. Held-out whole-trajectory coverage
was 87.5% (Wilson 95% CI [73.9%, 94.5%]), identical to run 1. No unsafe ACT
occurred, trivially, because no ACT occurred in either run. Paired
non-inferiority against estimator-side stopping FAILS its predeclared 5%
margin for all three tasks, by a *wider* margin than run 1 (because the
looser tolerances raised the estimator's own completion rate while the
certificate's completion rate stayed at zero). No latency claim is made
(`trajectory_prefix_estimate` mode).

This is the same falsifying result as run 1 under this repository's own
falsifier list in `RESEARCH_QUESTION.md` ("`k_C < k_E` is rare or absent";
"the observable proxy cannot support a useful calibrated envelope"), but this
run adds information run 1 could not provide on its own: it **rules out**
run 1's own leading hypothesis for the cause ("the observed 0% certificate
rate may reflect tolerances that are simply too tight ... rather than a
fundamental property of coverage-qualified stopping"). Tolerances were not
the bottleneck at this scope — the calibrated envelope width itself is. This
points the diagnosis at the observable feature vector / conformal-calibration
step (`cqts/safety.py` + the 12-dimensional feature vector in `config.json`)
rather than at the task-tolerance numbers, as the more likely place a future
run should look (a tighter or better-shaped observable proxy, more
calibration data, or a different nonconformity score), before trying a third
tolerance value.

## Physical robot extension

Not applicable. No physical manipulation was executed.

## What this run does and does not establish (H1-H4, `RESEARCH_QUESTION.md`)

- **H1** (a real backend can expose an observable proxy from which a calibrated
  completion envelope can be constructed without leaking ground truth):
  **same conclusion as run 1, now on firmer ground**. A real, non-ground-truth
  observable feature vector produces a finite (not +infinity) calibrated
  envelope, but that envelope is too loose to be discriminative at any of the
  two tolerance settings tried so far. Because tolerance was the only thing
  varied between the two runs and completion rate did not move, this run
  strengthens (does not newly establish, but corroborates) the reading that
  the bottleneck is the envelope-construction step itself, not an
  arbitrary/wrong tolerance choice. H1 remains: constructible, not yet shown
  useful/discriminative on real backend data.
- **H2** (reduces mean refinement stages vs. estimator stop): **not
  supported**, unchanged from run 1 — `k_C < k_E` never occurred (0/40 for
  every task, in both runs).
- **H3** (reduces end-to-end perception latency): **not tested**, unchanged
  from run 1 — `timing_mode=trajectory_prefix_estimate` deliberately disables
  a latency verdict.
- **H4** (task completion is non-inferior with HOLD in the denominator):
  **not supported**, and *more clearly rejected* than in run 1 for this
  backend/scope — the non-inferiority lower confidence bound moved further
  below the -5% margin in all three tasks (-49.1%, -43.9%, -23.7% vs run 1's
  -13.2%, -13.2%, -8.8%) because loosening tolerances raised the comparator's
  (estimator's) completion rate while the candidate's (certificate's) stayed
  at zero.

## Explicit limits of this run (do not read past these)

- In-house execution only; not independently replicated by another
  institution (same status as run 1, per `CONTRIBUTING.md`'s "Evidence
  status" section).
- Same segmentation, initial-pose-perturbation, and sampling-unit limitations
  as run 1 (dataset ground-truth `mask_visib` mask, synthetic bounded
  perturbation of ground truth for the initial pose, 20 independent test
  frames underlying 40 test episodes) — see run 1's RESULT.md for the full
  statement; unchanged here since the underlying data is byte-identical.
- This run varied exactly one factor (task tolerance) and held everything
  else fixed by design (see `TOLERANCE_DECISION.md`); it therefore cannot by
  itself diagnose *which part* of the envelope-construction pipeline
  (feature vector shape, conformal nonconformity score, calibration sample
  size, or something else) is responsible for the wide envelope — only that
  tolerance was not it, at this scope.
- The p75/1mm-1deg-rounding tolerance-derivation rule is one reasonable
  choice among several (p50 or p90 percentiles, different rounding grids)
  and was fixed before being run; a different reasonable choice might have
  produced somewhat different (though very unlikely dramatically different,
  given the envelope's width relative to tolerance) numbers. This is not
  re-run under multiple percentile choices in this cycle.
- No distribution-shift condition was run, same as run 1.
