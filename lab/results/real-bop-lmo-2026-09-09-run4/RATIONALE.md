# Predeclaration — real-bop-lmo-2026-09-09-run4 (PROP-DECAY-01)

Fourth real-data cycle. Founder authorization (`ops/HANDOFF_2026-09-09_external_dataset.md`,
"Fourth cycle authorized"): replace C6's fitted log-linear error-scale predictor
(`theory/FORMALIZATION_v1.md`) with a closed-form decay law derived from the ICP normal-equations
matrix's own conditioning, per Toledo proposal PROP-DECAY-01
(`~/ANSE.ASIA/toledo/registry/proposals/spectral_decay_predictor.json`).

## What is varied, what is not

Varied (the ONE deliberately changed factor): the error-scale predictor C6 is replaced by
`lab/decay_predictor.py`'s closed-form `s_{k,i} = rho_k^(K-k) * max(|proxy_i|, floor_i)`.

Held fixed, reused byte-identical from run2/run3:
- Dataset, frames, seeds, split (60 frames drawn seed 20260909, permuted seed 20260909, cut
  20/20/20 TRAIN/CALIBRATION/FINAL TEST) — confirmed identical `split_manifest.json` between
  `lab/data/bop_lmo_episodes/` (runs 1-3) and the newly generated `lab/data/bop_lmo_episodes_decay/`
  (run4; the only difference is one added `"decay"` field per stage, from
  `lab/adapter_bop.py:export_episode_decay`).
- ICP backend and its Kabsch update (`lab/bop_icp_backend.py`) — the ONLY backend change is a
  purely additive diagnostic, `normal_equations_H()` / `ICPStage.hessian`, which exposes
  `H_k = J^T J` (the same Jacobian `local_proxy()` already built internally); the Kabsch
  closed-form solve itself is untouched.
- Task tolerances, `error_floor`, `alpha=0.1`, inference config (margin/confidence/minimum_n):
  reused byte-identical from run2/run3's `config.json`.
- Calibration construction: **whole_trajectory (C7-C9)**, run1-2's original code, unmodified.

## Why whole_trajectory, not the run3 Bonferroni-checkpoint construction

Task instructions ask to combine the new predictor with whichever of the two prior calibration
variants gave the smaller `q` in run3's own results, and to justify the choice. Run3's own result
table: the four Bonferroni per-checkpoint quantiles (`q` in [2.085, 2.200]) were **all larger**
(4.3-10.0%) than the joint whole-trajectory `q ≈ 2.0008` used by runs 1-2. So: **whole_trajectory
gives the smaller `q`**, and this run combines PROP-DECAY-01's predictor with the unmodified
whole_trajectory (C7-C9) construction, not the Bonferroni-checkpoint one. `lab/run_real_system.py
--mode decay_predictor` implements exactly this pairing (new predictor, old whole-trajectory
calibration code, called via the same `nonconformity`/`conformal_quantile`/`evaluate` functions
runs 1-2 used, unmodified except for a documented `isinstance(model, DecayModel)` branch inside
`predicted_log_error()` that leaves the original fitted-regression branch untouched).

## The H_k ~ graph-Laplacian analogy: honest structural finding (from Task A + inspecting real H_k)

PROP-DECAY-01's own `honest_caveats` flagged the H_k~L_R identification as an assumption, not a
fact. Having now (a) exposed the real `H_k = J^T J` this backend computes and (b) attempted (in the
paired Toledo-repo task) to Coq-prove `q_formal/M.07`'s diameter floor `lambda_2 >= 4/(nD)`: **the
analogy does not hold structurally for this backend.** `H_k` is a fixed 6x6 SPD Gauss-Newton
Hessian over the 6 pose degrees of freedom; it has no vertex/edge structure, so a graph "diameter D"
is simply undefined for it, and its dimension never grows with the correspondence count `n`
(more inliers change its 6 eigenvalues quantitatively, never its shape). The natural quantity for
`H_k` is the ordinary matrix condition number `kappa = lambda_max/lambda_min` (its smallest of 6
eigenvalues, not a graph Fiedler value) feeding the classical (cited, not re-derived)
Gauss-Newton/Kantorovich contraction bound `rho <= (kappa-1)/(kappa+1)`, which needs no graph
structure and no `q_formal/M.07` bound at all. `q_formal/M.07`'s own Coq witness attempt (Task A,
Toledo repo) also did not close in the time given — see that repo's own report — so even had the
analogy held, no verified numeric floor exists yet to compare against. `lab/decay_predictor.py`
separately builds a real k-NN correspondence graph purely to report its own `n`/diameter honestly
(diagnostic only, reported in `RESULT.md`), confirming the dimensional mismatch numerically.

## No parameters tuned by peeking at final-test outcomes

`rho_k`/`kappa_k` are computed directly from `H_k`'s eigenvalues (`numpy.linalg.eigvalsh`), a
closed-form function with zero free parameters fit on any split. The only design choices made
before opening `test.jsonl` were: (1) use `H_k`'s ordinary condition number (not a graph bound,
since none holds/exists) — decided from the structural analysis above, before this run's numbers
were computed; (2) combine with whole_trajectory calibration — decided from run3's already-published
`q` comparison, not from this run's own numbers; (3) reuse run2/run3's tolerances/floor/alpha
unchanged.

## Commit discipline (F4, hardened since run2)

This file, `config.json`, and `split_manifest.json` are committed in their OWN commit, **before**
`lab/run_real_system.py --mode decay_predictor` is invoked against `test.jsonl` for this run.
`lab/data/bop_lmo_episodes_decay/{train,calibration,test}.jsonl` (deterministic, seed-driven,
regenerated by `lab/generate_bop_episodes.py --variant decay`) is committed alongside, since it
is itself part of the frozen input, not a final-test output.
