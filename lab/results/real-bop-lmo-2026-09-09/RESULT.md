# External Lab Result Report — real-bop-lmo-2026-09-09

Filled per `lab/RESULT_TEMPLATE.md`. This run was executed **in-house**, in this
repository, not by an independent laboratory. Do not read it as external
validation; read it as the repository's own first real-sensor step past the
numerical-only fixture (`experiments/numerical_icp.py`).

## Laboratory and system

- Institution: none (in-repo run, not an external lab)
- Laboratory: n/a
- Date: 2026-09-09
- Backend name: in-repo numpy+scipy point-to-point ICP/Kabsch
- Backend version / git commit: `ed8735ad04c17efe1d1bc0a0cec4625071d2bdbb`
- Model weights identifier: none (no learned model)
- Camera / sensor: BOP LM-O test scene 000002's own recorded RGB-D sensor (not our sensor; archived dataset)
- GPU / CPU: AMD Ryzen 7 4800H (CPU only; GTX 1650 Ti present but unused)
- Operating system: Linux 7.0.0-30-generic x86_64
- Object set: BOP LM-O `obj_id` 1 (ape), 8 (driller)
- Task set: `top_suction`, `label_alignment`, `keyed_insertion` (reused unchanged from `lab/config.example.json`)
- Ground-truth source: BOP LM-O `scene_gt.json` (dataset-provided independent pose annotation)
- SE(3) error convention: `lab/se3.py` (`T_error = inv(T_estimate) @ T_ground_truth`); ground-truth and estimated rotations re-orthonormalized via SVD before use (see `config.json`)

## Frozen protocol

- TRAIN episodes: 40 (20 frames x 2 objects)
- CALIBRATION episodes: 40 (20 frames x 2 objects)
- FINAL TEST episodes: 40 (20 frames x 2 objects)
- Split rule: 60 frames (with both objects present and each object's mask_visib region having >=300 depth-valid pixels) drawn without replacement from a 177-frame pool, then permuted and cut 20/20/20; see `config.json` and `split_manifest.json`
- `alpha`: 0.1
- nominal whole-trajectory coverage: 0.9
- non-inferiority margin: 0.05
- estimator-side stopping rule: two consecutive stages with |delta_t|<=0.5mm and |delta_r|<=0.1deg, else terminal stage
- online feature vector definition: see `config.json` (`observable_feature_vector`), 12-dimensional, ICP-residual/inlier/update/dispersion-proxy quantities only, no ground truth
- task reader definition: reused unchanged from `lab/config.example.json`
- deviations from `lab/README.md`: `timing_mode=trajectory_prefix_estimate` (no separately executed dual-policy timing was run, so no latency PASS/FAIL is claimed); `inference.independent_sampling_unit` is set to `episode` per the evaluator's only supported unit, while the dataset's *true* independent sampling unit is the frame (20 independent test frames, not 40 test episodes) — flagged explicitly in `config.json` as a real limitation of this run's statistical power, not resolved silently

## Primary results

Held-out FINAL TEST whole-trajectory envelope coverage: **87.5%** (32/40), Wilson 95% CI [73.9%, 94.5%] — the interval straddles the 90% nominal target; this run neither confirms nor rules out in-distribution coverage calibration at this sample size.

| Task | test coverage | `k_C<k_E` | mean `k_C-k_E` [95% CI] | estimator completion | certificate completion | completion difference [95% CI] | HOLD | unsafe ACT | perception latency difference [95% CI] | non-inferiority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| top_suction | 87.5% | 0.0% | n/a (no certificate ever fired) | 2.5% | 0.0% | -2.5 pp | 100% | 0.0% | +0.177 ms [0.000, 0.530] (endpoint-k-based prefix, not a speed claim) | FAIL (LCB -13.2% < -5% margin) |
| label_alignment | 87.5% | 0.0% | n/a | 2.5% | 0.0% | -2.5 pp | 100% | 0.0% | +0.177 ms [0.000, 0.530] | FAIL (LCB -13.2% < -5% margin) |
| keyed_insertion | 87.5% | 0.0% | n/a | 0.0% | 0.0% | 0.0 pp | 100% | 0.0% | +0.177 ms [0.000, 0.530] | FAIL (LCB -8.8% < -5% margin) |

The "perception latency difference" row is the paired trajectory-prefix-cost
difference in `trajectory_prefix_estimate` mode; per `lab/README.md` section 5
this is explicitly **not** a measured online-policy speedup, and
`latency_reduction_pass` is `null` in `lab_results.json` for every task.

## Mandatory failure accounting

- Number of unsafe ACTs on covered episodes: 0 (all three tasks)
- Number of unsafe ACTs total: 0 (all three tasks) — because the certificate never fired (`certificate_rate = 0.0` for all three tasks), there was no ACT to be unsafe
- Number of HOLD episodes: 40/40 for all three tasks (100%)
- Cases where `k_C >= k_E`: 40/40 (certificate never occurred, so `k_C` is undefined/+infinity for every episode)
- Calibration/test coverage shortfall, if any: FINAL TEST coverage 87.5%, 2.5 percentage points below the 90% nominal target; Wilson 95% CI [73.9%, 94.5%] includes the target, so a shortfall is plausible but not established at this sample size
- Distribution-shift conditions tested: none; this run is in-distribution only (no occlusion/lighting/object-swap shift condition was constructed)

## Claim form

On the in-repo numpy+scipy point-to-point ICP/Kabsch backend under BOP LM-O
scene 000002 (ape and driller, 40 held-out real-sensor test episodes, 15-iteration
budget), coverage-qualified stopping **did not** occur before estimator-side
stopping in any held-out trial for any of the three declared tasks
(`k_C < k_E` rate = 0%). The declared completion envelope, calibrated at
alpha=0.1 on 40 real-sensor calibration episodes, was too wide (calibration
quantile `q ≈ 2.0` in log-error units) ever to satisfy the fixed task
tolerances reused from the numerical fixture, so the certificate policy HOLD-ed
on 100% of test episodes for all three tasks and never emitted a single ACT.
Held-out whole-trajectory coverage was 87.5% (Wilson 95% CI [73.9%, 94.5%]).
No unsafe ACT occurred, trivially, because no ACT occurred. Paired
non-inferiority against estimator-side stopping FAILS its predeclared 5%
margin for all three tasks (the certificate's own zero-completion rate against
the estimator's near-zero-but-nonzero completion rate under Clopper-Pearson
bounds). No latency claim is made (`trajectory_prefix_estimate` mode).

This is a genuine falsifying result under this repository's own falsifier list
in `RESEARCH_QUESTION.md`: "`k_C < k_E` is rare or absent" and "the observable
proxy cannot support a useful calibrated envelope" both describe what was
observed here, for this backend/task/tolerance/budget combination.

## Physical robot extension

Not applicable. No physical manipulation was executed.

## What this run does and does not establish (H1-H4, `RESEARCH_QUESTION.md`)

- **H1** (a real backend can expose an observable proxy from which a calibrated
  completion envelope can be constructed without leaking ground truth):
  **partially supported operationally, not usefully**. A real, non-ground-truth
  observable feature vector was built from real ICP residuals/inliers/updates/
  dispersion, and `cqts/safety.py`'s split-conformal machinery did produce a
  finite calibrated envelope (not the +infinity fail-closed case). But the
  resulting envelope was too loose relative to the reused task tolerances to
  ever certify — so while an envelope *was* constructed, it was not, on this
  run, a *useful* one. H1 is not confirmed as generally true of real backends;
  it is shown constructible but non-discriminative here.
- **H2** (reduces mean refinement stages vs. estimator stop): **not
  supported** — `k_C < k_E` never occurred (0/40 for every task).
- **H3** (reduces end-to-end perception latency): **not tested** —
  `timing_mode=trajectory_prefix_estimate` deliberately disables a latency
  verdict, and the prefix-cost numbers that are reported are consistent with
  no benefit (certificate always ran to the terminal budget).
- **H4** (task completion is non-inferior with HOLD in the denominator):
  **not supported** — non-inferiority FAILS its predeclared 5% margin for all
  three tasks, driven by the certificate's 0% completion (all HOLD) against
  the estimator's own near-zero completion.

## Explicit limits of this run (do not read past these)

- In-house execution only; not independently replicated by another
  institution (`CONTRIBUTING.md`'s "Evidence status" section applies:
  maintainers do not label this as independently replicated).
- Object segmentation used the dataset's ground-truth `mask_visib` mask, not
  an online detector; this tests refinement only.
- The initial pose came from a synthetic bounded perturbation of ground truth,
  not a real coarse-pose detector or prior frame's tracked estimate.
- 20 independent test frames (40 test episodes formed from 2 objects per
  frame) is a small sample; the coverage Wilson interval is wide, and the
  non-inferiority bound is conservative by construction (Clopper-Pearson +
  Bonferroni) at this n.
- Task tolerances were reused unchanged from the numerical fixture and were
  not re-derived from the real geometric/task requirements of the ape/driller
  objects; the observed 0% certificate rate may reflect tolerances that are
  simply too tight for this backend's real achievable accuracy within 15
  iterations, rather than a fundamental property of coverage-qualified
  stopping. This is exactly the kind of negative result `CONTRIBUTING.md`
  asks to be reported rather than hidden or re-tuned post hoc.
- No distribution-shift condition was run (unlike the numerical fixture's
  `stress_2x_sensor_noise` ablation).
