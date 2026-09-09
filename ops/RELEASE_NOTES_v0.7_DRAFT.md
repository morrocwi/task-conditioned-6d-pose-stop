# Release notes (draft) — v0.7: first real-sensor lab result (falsifying)

Status: draft for the chair's review. Not tagged, not pushed to origin, no GitHub release created by this workflow.

## What this release is

The first real-sensor run of this repository's lab harness (`lab/run_real_system.py`,
`cqts/safety.py`) against a genuine external dataset: BOP LineMOD-Occlusion (LM-O),
objects `ape` and `driller`, 60 real RGB-D frames drawn into 120 episodes and split
40/40/40 TRAIN/CALIBRATION/FINAL TEST by frame. The backend is a from-scratch
numpy+scipy point-to-point ICP/Kabsch implementation wired through
`lab/ADAPTER_TEMPLATE.py`; there is no learned model and no neural component.

This is **in-house execution, not independent-lab validation**. It is a step past the
numerical-only fixture (`experiments/numerical_icp.py`) onto real depth-derived point
clouds and real dataset-provided ground truth, nothing more. Full accounting, raw
numbers, and provenance: `lab/results/real-bop-lmo-2026-09-09/RESULT.md`,
`config.json`, `split_manifest.json`, `backend_and_hardware.md`.

## The result is a falsifying result, plainly

For this specific backend/task/tolerance/budget combination, the certificate never
fired. `k_C < k_E` — the core computational opportunity this repository exists to
test — occurred on **0% of held-out test episodes**, for all three declared tasks.
The calibrated completion envelope was finite (the split-conformal machinery did not
fail closed to +infinity), but it was too wide relative to the reused task tolerances
ever to certify, so the policy HOLD-ed on 100% of test episodes and never emitted a
single ACT. Because no ACT occurred, the unsafe-ACT rate is trivially 0% — this is
not a safety success to advertise, it is the arithmetic consequence of the certificate
never firing.

Held-out whole-trajectory coverage was 87.5% (35/40), Wilson 95% CI [73.9%, 94.5%],
which straddles the 90% nominal target — this run neither confirms nor rules out
in-distribution coverage calibration at this sample size.

| Task | test coverage | `k_C<k_E` | estimator completion | certificate completion | HOLD | unsafe ACT | non-inferiority |
|---|---:|---:|---:|---:|---:|---:|---|
| top_suction | 87.5% | 0.0% | 2.5% | 0.0% | 100% | 0.0% | FAIL (LCB -13.2% < -5% margin) |
| label_alignment | 87.5% | 0.0% | 2.5% | 0.0% | 100% | 0.0% | FAIL (LCB -13.2% < -5% margin) |
| keyed_insertion | 87.5% | 0.0% | 0.0% | 0.0% | 100% | 0.0% | FAIL (LCB -8.8% < -5% margin) |

(No latency PASS/FAIL is claimed: `timing_mode=trajectory_prefix_estimate` deliberately
disables a latency verdict; see `RESULT.md` for the full table including the
perception-latency-difference column.)

## What this does and does not establish, against `RESEARCH_QUESTION.md` H1-H4

- **H1** (a real backend can expose an observable proxy from which a calibrated
  completion envelope can be constructed without leaking ground truth): partially
  supported operationally, not usefully. A real, non-ground-truth observable feature
  vector was built and a finite envelope was produced — but it was too loose relative
  to the reused task tolerances to ever certify. Shown constructible, not shown
  discriminative, on this run.
- **H2** (reduces mean refinement stages vs. estimator stop): **not supported** —
  `k_C < k_E` never occurred (0/40 for every task).
- **H3** (reduces end-to-end perception latency): **not tested** — timing mode
  deliberately disables a latency verdict.
- **H4** (task completion is non-inferior with HOLD in the denominator): **not
  supported** — non-inferiority fails its predeclared 5% margin for all three tasks.

This is a genuine falsifying case under this repository's own falsifier list in
`RESEARCH_QUESTION.md`: "`k_C < k_E` is rare or absent" and "the observable proxy
cannot support a useful calibrated envelope" both describe what was observed here,
for this backend/task/tolerance/budget combination. It does not refute the research
question in general — see `RESULT.md`'s "Explicit limits" section: task tolerances
were reused unchanged from the numerical fixture rather than re-derived for the real
ape/driller objects, so the 0% certificate rate may reflect tolerances too tight for
this backend's real achievable accuracy in 15 iterations, not a fundamental property
of coverage-qualified stopping. Sample size is also small (20 independent test
frames), and segmentation/initial-pose used dataset ground truth and a synthetic
perturbation rather than an online detector or tracker.

## Process note carried forward, not hidden

The run's predeclared task tolerances, alpha, error floor, and inference fields are
byte-identical to `lab/config.example.json`, committed well before this task, so the
values that gate certification could not have been tuned to this run's outcome. The
frame-sampling, 300px depth-validity prefilter, and split-shuffle parameters specific
to this run, however, were committed in the same commit as the results
(`config.json` and `split_manifest.json` were added alongside `lab_results.json`),
so that part of the freeze is asserted in the writeup rather than independently
git-verifiable against an earlier commit. This does not by itself change the reported
numbers — they were independently recomputed and matched — but it is a process gap
this repository's `CONTRIBUTING.md` freeze discipline should close for future runs:
commit the predeclared per-run config/split rule before generating or inspecting any
test-split output.

## Plain-language summary (for the GitHub release body)

We ran the lab's task-conditioned stopping method for the first time on real camera
data instead of synthetic numbers — using a public benchmark dataset (BOP
LineMOD-Occlusion) with two real objects and 60 real depth photos. The result is
negative: for this backend and these task tolerances, the method's "early stop"
certificate never fired even once across 40 held-out test cases, so it never
recommended stopping refinement early. That means we cannot show, with this run,
that the method saves computation or time on real sensor data. It is not a safety
failure (nothing unsafe happened, because nothing happened) and it does not prove the
underlying idea is wrong — the task tolerances reused here were not re-tuned for
these particular real objects, and the test set is small — but it is an honest,
reported failure to confirm the method's benefit on this first real-data attempt, and
we are publishing it as such rather than reframing it as a success.

## Verdict

RELEASE READY: yes
