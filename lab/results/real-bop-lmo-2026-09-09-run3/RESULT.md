# External Lab Result Report — real-bop-lmo-2026-09-09-run3

Third real-data cycle. Tests the PROP-CONF-03 Bonferroni-corrected multi-checkpoint conformal
band against the diagnosis in `~/ANSE.ASIA/glosa/projects/GLS-2026-005_pose-stop-conformal-diagnosis/DIAGNOSIS_HYPOTHESIS.md`.
Predeclaration commit `681c710` (2026-09-09T14:32:42+07:00, before test.jsonl was evaluated for
this run's frozen config — see `RATIONALE.md` for the exact predeclaration and its disclosed
process note). Evaluator invoked 2026-09-09T14:32:49+07:00.

## Laboratory and system

- Institution: in-house (not an independent external laboratory — same as runs 1-2)
- Date: 2026-09-09
- Backend name: in-repo numpy+scipy point-to-point ICP/Kabsch (`lab/bop_icp_backend.py`)
- Backend version / git commit: unchanged since run 1/run2; this run's own code commit `cfe50a4`
- Model weights identifier: none (no learned model)
- Camera / sensor: BOP LM-O `test_bop19` real RGB-D, scene `000002`
- GPU / CPU: AMD Ryzen 7 4800H (CPU-only; GPU present, unused)
- Operating system: Linux 7.0.0-30-generic x86_64
- Object set: `obj_id=1` (ape), `obj_id=8` (driller)
- Task set: `top_suction`, `label_alignment`, `keyed_insertion` (unchanged definitions)
- Ground-truth source: BOP `scene_gt.json`, orthonormalized via SVD
- SE(3) error convention: `[tx,ty,tz,rx,ry,rz]`, metres/radians

## Frozen protocol

- TRAIN episodes: 40 (byte-identical to runs 1-2)
- CALIBRATION episodes: 40 (byte-identical to runs 1-2)
- FINAL TEST episodes: 40 (byte-identical to runs 1-2)
- Split rule: unchanged — 60 frames drawn without replacement, seed 20260909, cut 20/20/20
- `alpha`: 0.1 (same as runs 1-2)
- nominal whole-trajectory coverage: 0.9
- non-inferiority margin: 0.05
- estimator-side stopping rule: unchanged (two consecutive ICP stages within 0.5mm/0.1deg)
- online feature vector definition: unchanged (12-dim, see run2's config.json)
- task reader definition: unchanged (box/l1 masks and tolerances, reused from run2 — TRAIN-derived)
- deviations from `lab/README.md`: certificate construction is the NEW PROP-CONF-03
  Bonferroni-corrected multi-checkpoint band (`lab/multicheckpoint.py`), checked ONLY at
  checkpoints `k in {4, 8, 12, 15}` (K'=4), each calibrated at `alpha/K'=0.025` via the same
  `cqts.safety.safe_split_conformal_quantile` used by the whole-trajectory path, instead of the
  single joint whole-trajectory max-over-15-stages construction used by runs 1-2. This is the one
  deliberately varied factor.

## The falsifiable prediction and its outcome

GLS-2026-005's diagnosis predicted: if whole-trajectory max-aggregation over 15 stages is the
dominant driver of runs 1-2's ~7.4x envelope inflation (joint `q ≈ 2.0008`), then the
per-checkpoint Bonferroni-corrected quantiles should be SMALLER than the joint `q`, and the
certificate rate should rise above 0%.

**Result: the prediction is REFUTED.** All four per-checkpoint quantiles are LARGER than the joint
`q ≈ 2.0008`, not smaller, and the certificate rate remains exactly 0.0 on all three tasks,
identical to runs 1-2.

| Checkpoint k | q_{alpha/K'} (this run) | vs. joint q≈2.0008 (runs 1-2) | calibration rank | held-out checkpoint coverage |
|---:|---:|---|---:|---:|
| 4  | 2.0854 | +4.3% larger | 40/40 | 97.5% (Wilson 87.1-99.6%) |
| 8  | 2.1715 | +8.5% larger | 40/40 | 100% (Wilson 91.2-100%) |
| 12 | 2.1636 | +8.1% larger | 40/40 | 100% (Wilson 91.2-100%) |
| 15 | 2.2000 | +10.0% larger | 40/40 | 100% (Wilson 91.2-100%) |

"All-checkpoints-covered" rate (the union-bound event: every one of the 4 checkpoints
simultaneously covered on the same episode): 97.5% (Wilson 87.1-99.6%), n=40.

**Why this happened (diagnosed after the fact, `Dr` tier, not yet independently re-verified):**
removing the 15-way stage-max did shrink the *aggregation* term, but each per-checkpoint
calibration rank is `ceil(41*(1-0.025)) = 40` — the SINGLE LARGEST of the 40 calibration scores,
i.e. `rank=n` exactly, the most extreme order statistic short of the fail-closed `+infinity`
boundary. The joint whole-trajectory construction's rank was `37/40`, comfortably inside the
calibration sample. Splitting `alpha` four ways (`alpha/K'=0.025`) pushed every checkpoint's
required rank to the very top of only 40 calibration episodes, and the top-of-sample score
happened to be large enough at every checkpoint that the Bonferroni correction's own
conservativeness (needed for the union bound to be honest) outweighed the removed
stage-aggregation term. This is exactly the standard tradeoff the union bound makes explicit: K'
independent guarantees at level `alpha/K'` are only cheaper than one joint guarantee at level
`alpha` when the *stage-aggregation* penalty being removed is larger than the *per-checkpoint
Bonferroni* penalty being added — with only n=40 calibration episodes, it was not larger here.

## Primary results

| Task | test coverage (all-checkpoints) | `k_C<k_E` | mean `k_C-k_E` [95% CI] | estimator completion | certificate completion | completion difference [95% CI] | HOLD | unsafe ACT | perception latency difference [95% CI] | non-inferiority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| top_suction | 97.5% | 0/40 | +0.05 [0.0, 0.15] ms-equivalent k | 32.5% | 0.0% | -32.5 pp [LCB -49.1%] | 100% | 0/40 | +0.18ms [0.0, 0.53] | FAILS |
| label_alignment | 97.5% | 0/40 | +0.05 [0.0, 0.15] | 27.5% | 0.0% | -27.5 pp [LCB -43.9%] | 100% | 0/40 | +0.18ms [0.0, 0.53] | FAILS |
| keyed_insertion | 97.5% | 0/40 | +0.05 [0.0, 0.15] | 10.0% | 0.0% | -10.0 pp [LCB -23.7%] | 100% | 0/40 | +0.18ms [0.0, 0.53] | FAILS |

(`k_C-k_E` and the latency row are structurally uninformative here: with certificate_rate=0.0
across the board, every "certificate" endpoint is the terminal stage k=15, so the `k_C<k_E` count
is mechanically 0/40 for all three tasks, same as runs 1-2.)

## Mandatory failure accounting

- Number of unsafe ACTs on covered episodes: 0 (mechanically impossible: certificate never fires)
- Number of unsafe ACTs total: 0
- Number of HOLD episodes: 40/40 for all three tasks (100%)
- Cases where `k_C>=k_E`: 40/40 for all three tasks (certificate never fires, so k_C is undefined /
  treated as the terminal stage, which is >= k_E for every episode)
- Calibration/test coverage shortfall: none observed at the per-checkpoint or all-checkpoints
  level (97.5-100% against a 90% nominal target, i.e. over-covered, not under-covered — consistent
  with q's being even larger than the already-conservative joint construction)
- Distribution-shift conditions tested: none (same in-distribution TRAIN/CALIBRATION/TEST split as
  runs 1-2)

## Claim form

On the in-repo numpy+scipy point-to-point ICP/Kabsch backend under real BOP LM-O depth data
(ape/driller, scene 000002), the PROP-CONF-03 Bonferroni-corrected multi-checkpoint conformal
construction did not certify on any of 40 held-out trials for any of three declared tasks
(certificate rate 0%, HOLD 100%), and its per-checkpoint calibrated quantiles were 4.3-10.0% LARGER
than the whole-trajectory joint quantile used by runs 1-2, not smaller as the tested hypothesis
predicted. Non-inferiority against estimator-side stopping fails its predeclared 5% margin for all
three tasks (identical lower confidence bounds to run 2, since the certificate policy's completion
outcomes are mechanically identical whenever certificate_rate=0.0 in both constructions).

## What this does and does not mean

This REFUTES the specific falsifiable prediction recorded in GLS-2026-005 (that restricting
aggregation to K'<=4 predeclared checkpoints, coordinate-max only, would shrink `q_alpha` below the
joint whole-trajectory value and raise the certificate rate above 0%). It does **not** show the
PROP-CONF-03 construction itself is wrong: the union-bound argument it composes with is
machine-checked (`toledo/coq/canonical/PROP_CONF_03_union_bound.v`), and the coverage numbers
observed here (97.5-100% all-checkpoints-covered) are consistent with, not a violation of, the
`>=90%` guarantee it is designed to provide. What is refuted is the more specific *quantitative*
prediction that this particular aggregation restriction would be numerically CHEAPER than the
joint construction at this specific sample size (n=40). The mechanism now looks more precisely
located: at n=40 calibration episodes, the Bonferroni per-checkpoint penalty (each checkpoint
needing the top-of-sample order statistic) is at least as large as the stage-aggregation penalty
it was meant to remove. A larger calibration sample size (which would push each checkpoint's
required rank away from the very top of the sample) is the natural next lever this result points
to, not yet tested here.

This closes P08 discipline 2 (test the prediction) for the hypothesis recorded in GLS-2026-005:
the prediction was tested on real data and refuted. Discipline 4 (independent re-verification by a
materially separate pass) is still not yet applied to this new result, same caveat as before.

## Physical robot extension

Not applicable — no physical manipulation was executed in this cycle, same as runs 1-2.
