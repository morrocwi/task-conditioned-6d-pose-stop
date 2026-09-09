# External Lab Result Report — real-bop-lmo-2026-09-09-run4

Fourth real-data cycle. Tests PROP-DECAY-01 (`~/ANSE.ASIA/toledo/registry/proposals/spectral_decay_predictor.json`):
replace C6's fitted log-linear error-scale predictor with a closed-form decay predictor derived
from the ICP normal-equations Hessian `H_k = J^T J`'s own conditioning. Predeclaration commit
`724c47a` (before `test.jsonl` was evaluated for this run — see `RATIONALE.md`).

## Laboratory and system

- Institution: in-house (not an independent external laboratory — same as runs 1-3)
- Date: 2026-09-09
- Backend name: in-repo numpy+scipy point-to-point ICP/Kabsch (`lab/bop_icp_backend.py`), plus one
  additive diagnostic (`normal_equations_H`, `ICPStage.hessian`) exposing `H_k` — the Kabsch
  closed-form update itself is unchanged from runs 1-3.
- Model weights identifier: none (predictor is closed-form, zero free parameters fit on TRAIN)
- Camera / sensor: BOP LM-O `test_bop19` real RGB-D, scene `000002` (identical to runs 1-3)
- Object set: `obj_id=1` (ape), `obj_id=8` (driller)
- Task set: `top_suction`, `label_alignment`, `keyed_insertion` (unchanged definitions/tolerances)
- Ground-truth source: BOP `scene_gt.json`, orthonormalized via SVD

## Frozen protocol

- TRAIN/CALIBRATION/FINAL TEST: 40/40/40 episodes, byte-identical split (same 60 frames, same
  seed 20260909 draw+shuffle) to runs 1-3 — confirmed by identical `split_manifest.json` contents
  between `lab/data/bop_lmo_episodes/` and the newly generated `lab/data/bop_lmo_episodes_decay/`.
- `alpha`=0.1, error_floor, task tolerances, inference config: reused byte-identical from run2/run3.
- Calibration construction: **whole_trajectory (C7-C9)**, unmodified code from runs 1-2 — chosen
  because run3 found it gives the smaller `q` (see `RATIONALE.md`).
- The one varied factor: C6 replaced by `lab/decay_predictor.py`'s
  `s_{k,i} = rho_k^{K-k} * max(|proxy_i|, floor_i)`, `rho_k = (kappa_k-1)/(kappa_k+1)`,
  `kappa_k = lambda_max(H_k)/lambda_min(H_k)` (ordinary 6-eigenvalue matrix condition number).

## Finding 1 (structural): the H_k ~ graph-Laplacian analogy is REFUTED for this real backend

PROP-DECAY-01 proposed reading `H_k` as an instance of the weighted graph-Laplacian family
`L_R := D_W - W`, so that `q_formal/M.07`'s diameter floor `lambda_2 >= 4/(nD)` (a Coq witness for
which was separately attempted in the paired Toledo-repo task — it did not close; see that task's
own report) could lower-bound `H_k`'s own `lambda_2`. Having now inspected the real `H_k` this
backend computes:

| quantity | `H_k` (ICP normal-equations Hessian) | correspondence graph (constructed separately, diagnostic only) |
|---|---|---|
| shape | fixed 6x6 (one row/col per pose DOF), **every stage, every episode** | grows with correspondence count |
| example (stage 0, episode `bopLMO-train-f000027-o1`) | `lambda_min=0.1235`, `lambda_max=985.7`, `kappa=7978`, 6 eigenvalues total | k-NN graph (k=6) over the 400 corresponded points: `n=400`, diameter `D=16` |
| "`lambda_2`" meaning | 2nd-largest of only 6 eigenvalues — no trivial null constant-vector eigenvalue exists, so there is no Fiedler-style "2nd smallest after 0" reading | 2nd-smallest Laplacian eigenvalue of an n-node graph (the actual Fiedler value `q_formal/M.07` is about) |
| `4/(n*D)` computed from this same stage's graph | **0.000625** | — |
| compares to `H_k`'s own `lambda_min` | 0.1235 — a ~200x difference, but this is not a meaningful comparison: the two numbers are eigenvalues of two *different, unrelated* matrices | |

`H_k` never grows past 6x6 no matter how many correspondences contribute to it (they only change
its 6 eigenvalues quantitatively); a graph "diameter" is undefined for a dense 6x6 Gram matrix with
no vertex/edge structure. **This confirms, numerically and not just by argument, that applying
`q_formal/M.07`'s bound to `H_k` is a category error for this real backend** — exactly the risk
PROP-DECAY-01's own `honest_caveats` flagged as an unproven assumption. The correct quantity for
`H_k` is the ordinary matrix condition number, which needs no graph object at all — that is what
this run actually uses (see `RATIONALE.md`).

## Finding 2 (empirical): the decay predictor does not fix, and slightly worsens, the calibration bottleneck

| | run1 (fixture tolerances) | run2 (TRAIN-derived tolerances) | run3 (Bonferroni checkpoints) | **run4 (decay predictor)** |
|---|---:|---:|---:|---:|
| certificate construction | whole_trajectory | whole_trajectory | Bonferroni multi-checkpoint | whole_trajectory |
| error-scale predictor | C6 log-linear | C6 log-linear | C6 log-linear | **PROP-DECAY-01 decay** |
| calibrated `q` (whole-trajectory joint, or per-checkpoint range) | ≈2.0008 | ≈2.0008 | [2.085, 2.200] | **3.298** |
| calibration rank | 37/40 | 37/40 | 40/40 (all 4 checkpoints) | 37/40 |
| held-out coverage | ~87.5% | ~87.5% | 97.5-100% per checkpoint | **95.0%** [Wilson 83.5-98.6%] |
| certificate_rate (all 3 tasks) | ~0-2.5% | 0.0% | 0.0% | **0.0%** |
| HOLD rate (all 3 tasks) | ~97.5-100% | 100% | 100% | **100%** |
| unsafe-ACT rate | 0% (by fail-closed construction) | 0% | 0% | 0% |
| non-inferiority pass | fails | fails (LCB -49.1%/-43.9%/-23.7%) | fails | fails (LCB -49.1%/-43.9%/-23.7%, identical to run2 — expected: estimator-side completion depends only on the shared ICP trajectories/tolerances, not on the certificate predictor) |

**The falsifiable prediction (a physically-derived decay law would give a tighter, better-behaved
calibrated envelope than the ad hoc fitted log-linear model) is REFUTED.** The calibrated `q` this
run produces (3.298, i.e. `exp(3.298)≈27x` multiplicative envelope inflation) is *larger* than both
the original C6 log-linear joint `q≈2.0008` (`≈7.4x` inflation, runs 1-2) and every one of run3's
Bonferroni per-checkpoint quantiles (2.085-2.200). Certificate rate is unchanged at exactly 0.0% on
all three tasks — the fourth cycle in a row to find this. Held-out coverage (95.0%, Wilson
[83.5%, 98.6%]) does comfortably clear the 90% target, consistent with `q`'s larger size (a wider
envelope over-covers).

**Likely cause (diagnosis, not yet independently re-verified — Dr tier at most):** the measured
`rho_k` values across all 640 (train-split) stage-episodes are tightly clustered near 1
(median 0.9991, range [0.9964, 0.9997]; `kappa_k` ranges roughly 557-7979). A `rho_k` this close to
1 makes the decay factor `rho_k^{K-k}` barely shrink the predicted error scale across the 15-stage
trajectory, so — unlike C6's regression, which at least fits some decreasing trend against TRAIN
labels — this closed-form predictor stays close to a near-constant multiple of the raw proxy at
every stage. Combined with C7's whole-trajectory max-over-stages-and-axes nonconformity score, a
predictor with little cross-stage shrinkage inflates the calibrated quantile rather than tightening
it. A likely contributor to `rho_k`'s near-1 clustering: `H_k`'s six eigenvalues span rotation
columns (`-skew(R p_i)`, scaled by the object's ~0.05-0.26m point-coordinate magnitudes) and
translation columns (identity-scaled) with very different natural units, so `kappa_k` (and hence
`rho_k`) is dominated by this coordinate-scaling artifact of the [omega, t] parameterization, not
by the geometric well/ill-conditioning of the correspondence set. This is a plausible, disclosed,
NOT independently re-verified explanation for why the closed-form predictor under-performs here.

## Positioning (unchanged guardrails)

This is real BOP LM-O sensor data through a real (non-learned) point-to-point ICP backend,
in-house, not independent external-lab validation. No claim is made that this run "solves 6D pose
estimation" or that any construction here is the first of its kind — see
`ops/HANDOFF_2026-09-09_external_dataset.md`'s novelty/positioning constraints, unchanged and
still binding. `runs 1-3`'s own HOLD/evidence-ceiling language is not retired by this run — if
anything, this run adds a fourth independent confirmation of the same calibration bottleneck, now
also across an entirely different (non-regression) predictor family.

## Test suite, leak-scan, push status

See the commit(s) following this report and the top-level task response for details.
