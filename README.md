# Coverage-Qualified Task Stopping for Iterative 6D Pose Refinement

Public, implementation-first research repository.

> **Single question:** can iterative 6D pose refinement stop before estimator-side convergence because the **entire calibrated pose-error completion set** is already acceptable for the declared downstream task?

The certificate hitting time is

\[
k_C=\inf\{k:\widehat{\mathcal C}^{cal}_k\subseteq\mathcal A_T\},
\]

and the estimator-side stopping stage is `k_E`.

The central event is

```text
k_C < k_E
```

If no certificate exists within budget, `k_C` is undefined / `+infinity`. The executed policy may continue to the terminal budget and return HOLD; that endpoint is **not** relabelled as `k_C`.

## Core distinction

```text
pose not known exactly
        !=
task verdict unresolved
```

The online rule is

```text
observable refinement readout
        -> calibrated completion envelope
        -> robust downstream task reader
        -> ACT | CONTINUE | HOLD
```

ACT is permitted only when every pose error retained by the envelope passes the declared task reader.

## What changed after adversarial review

The review of commit `0c7532d` found real correctness and inference defects. The repository was hardened rather than defending them.

Current corrections include:

- **finite split-conformal boundary fixed:** when `ceil((n+1)(1-alpha)) = n+1`, the augmented order statistic is `+infinity`; calibration becomes uninformative and the system HOLDs;
- **fail-closed numerics:** NaN, invalid model state, malformed task tolerances, or overflow can no longer collapse to a zero-width certificate;
- **paired binary inference:** the lab harness no longer uses a degenerate percentile bootstrap to certify non-inferiority; it uses paired discordances with a conservative finite-sample lower bound and requires a predeclared margin, confidence level, minimum N, and sampling unit;
- **timing claim separated:** trajectory-prefix cost is descriptive only. A speed claim requires separately executed online-policy timing including feature, gate, update, synchronization, and other policy-specific overhead;
- **certificate semantics corrected:** no-certificate episodes have no finite `k_C`; terminal HOLD is an executed endpoint, reported separately;
- **repetition claim narrowed:** the ten-cycle numerical study is explicitly a reduced-configuration diagnostic, not a repetition of the first full standard configuration;
- **matched comparator experiment added:** full budget, default estimator stop, risk-checked fixed stage, risk-checked estimator threshold, risk-checked learned scalar threshold, raw envelope, and trajectory-conformal certificate are evaluated on the same frozen splits with new comparison-test seeds.

See [`evidence/ADVERSARIAL_REVIEW_RESPONSE.md`](evidence/ADVERSARIAL_REVIEW_RESPONSE.md).

## Shared correctness core

Critical certificate and inference logic is centralized in [`cqts/safety.py`](cqts/safety.py), including:

```text
safe split-conformal order statistic
fail-closed error-bound construction
fail-closed task reader
Clopper-Pearson binomial bounds
paired binary non-inferiority lower bound
```

Adversarial regression tests retain the specific small-n, NaN, and n=1 inference counterexamples that motivated the revision.

## Evidence layers

1. **Finite diagnostic** — exhaustive toy routing fixture (`benchmark.py`).
2. **Numerical 6D backend** — iterative point-to-point ICP/Kabsch on generated 3-D point clouds (`experiments/numerical_icp.py`).
3. **Raw-proxy calibrated envelope** — high-coverage but deliberately retained negative control (`experiments/conformal_completion.py`).
4. **Learned-shape trajectory-conformal certificate** — TRAIN → CALIBRATION → FINAL TEST (`experiments/learned_conformal_completion.py`).
5. **Matched comparator study** — same-split, calibration-tuned alternatives on new test seeds (`experiments/matched_baselines.py`).
6. **Reduced-configuration repeated diagnostic** — repeated TRAIN → CALIBRATION → TEST cycles under a smaller computational configuration (`experiments/replication_study.py`).
7. **University lab harness** — backend-agnostic real-system interface (`lab/`).
8. **Real-sensor ICP run (in-house, not independent)** — real BOP LM-O RGB-D data through a real numpy+scipy point-to-point ICP/Kabsch backend, via the `lab/` harness (`lab/results/real-bop-lmo-2026-09-09/`). This is a negative/falsifying result: see "Real-sensor evidence" below.
9. **Second real-sensor ICP run, tolerance-rederivation cycle (in-house, not independent)** — same backend/dataset/split as run 8, task tolerances re-derived from TRAIN-split achievable accuracy instead of reused from the numerical fixture (`lab/results/real-bop-lmo-2026-09-09-run2/`). Same negative/falsifying result, and it rules out "the tolerances were just too tight" as the explanation: see "Real-sensor evidence" below.

The numerical research evidence uses generated point clouds. No camera, neural pose estimator, contact physics, or physical robot is silently implied by those results.

## Real-sensor evidence (in-house, not independent lab validation)

`lab/results/real-bop-lmo-2026-09-09/` runs the public `lab/` harness against real
RGB-D data (BOP LineMOD-Occlusion, scene 000002, objects ape and driller) and a real
in-repo numpy+scipy point-to-point ICP/Kabsch backend (`lab/bop_icp_backend.py`) —
no neural pose model, no learned weights. Object segmentation used the dataset's own
ground-truth `mask_visib` mask (this tests the refinement stage only, not detection),
and the ICP initial pose came from a documented bounded random perturbation of ground
truth (not a real coarse detector). This was executed **in-house, not by an
independent laboratory** — it does not upgrade the evidence tier described in
`CONTRIBUTING.md`'s "Evidence status" section.

Result: on 40 held-out real-sensor test episodes at 15-iteration budget, the
declared completion envelope, calibrated at `alpha=0.1` on real ICP residual/
inlier/update features, **never once** collapsed tight enough to satisfy the
task tolerances reused from the numerical fixture: `certificate_rate = 0.0` and
`k_C < k_E` on 0/40 episodes for all three declared tasks. Held-out
whole-trajectory coverage was 87.5% (Wilson 95% CI [73.9%, 94.5%]). Paired
non-inferiority against estimator-side stopping FAILS its predeclared 5%
margin for every task. No unsafe ACT occurred, but only because no ACT ever
occurred (100% HOLD). Full accounting, exact numbers, and explicit limits
(small n, reused untuned tolerances, no distribution-shift condition) are in
[`lab/results/real-bop-lmo-2026-09-09/RESULT.md`](lab/results/real-bop-lmo-2026-09-09/RESULT.md).

This is a genuine falsifying result under this repository's own falsifier list
in `RESEARCH_QUESTION.md` ("`k_C < k_E` is rare or absent"; "the observable
proxy cannot support a useful calibrated envelope") for this specific
backend/task/tolerance/budget combination. It does not retire any existing
OPEN/HOLD item in the Claim boundary section below — real RGB-D data was used,
but no real neural pose backend, no independent lab, no GPU/online-policy
timing, and no physical robot were involved.

### Second real-sensor cycle: re-deriving tolerances from TRAIN

`lab/results/real-bop-lmo-2026-09-09-run2/` re-runs the identical dataset,
split, and ICP backend as above, changing exactly one factor: task
tolerances were re-derived from the 75th percentile of TRAIN-split
achievable ICP accuracy (never from calibration/test data) instead of reused
from the numerical fixture — see
[`TOLERANCE_DECISION.md`](lab/results/real-bop-lmo-2026-09-09-run2/TOLERANCE_DECISION.md)
for the decision and reproducible derivation
(`lab/derive_tolerances_from_train.py`). The frozen config and split were
committed (`d9ac505f40df81344ffe0e3737cb1b51d36bf13e`) before FINAL TEST was
opened, fixing the F4 process gap disclosed in the v0.7.0 release notes.

Result: unchanged. Even with tolerances loosened 2.6-2.8x (translation) and
1.4-1.8x (rotation) — which did raise the *estimator's own* completion rate
substantially — the certificate's completion rate stayed at exactly 0% for
all three tasks, and `k_C < k_E` remained 0/40. Held-out coverage (87.5%)
was identical to run 1, since it depends on the calibration data and
observable-feature model, not on task tolerances. This rules out run 1's own
leading hypothesis ("tolerances were simply too tight") and points the
diagnosis at the calibrated-envelope-construction step (observable feature
vector / conformal calibration) rather than at the tolerance numbers. Full
accounting in
[`lab/results/real-bop-lmo-2026-09-09-run2/RESULT.md`](lab/results/real-bop-lmo-2026-09-09-run2/RESULT.md).

## First frozen learned-shape final test

The original full numerical protocol used:

```text
TRAIN        160 episodes
CALIBRATION  160 episodes
FINAL TEST   120 episodes
STRESS        60 episodes (2x sensor noise)
max stage     20
points       120
nominal marginal whole-trajectory coverage = 90%
```

Observed held-out whole-trajectory coverage was **85.00%** (Wilson 95% CI 77.53%–90.30%). It is below the nominal target and remains reported as such.

On that frozen numerical test, certificates occurred before estimator stopping in **90.0%**, **87.5%**, and **87.5%** of all assigned episodes for top suction, label alignment, and keyed insertion respectively. Overall certificate-policy completion differed from estimator stopping by **-2.50**, **-0.83**, and **0.00 percentage points**; observed unsafe ACT was zero in that 120-episode fixture. Zero observed events is not zero population risk.

The current code reports certificate hitting time separately from the executed terminal endpoint for HOLD episodes. Historical tables that called the budget-capped endpoint “mean `k_C`” should not be interpreted as a mean certificate hitting time.

## Negative control: coverage alone is not enough

The raw-proxy conformal construction achieved **95.83%** held-out whole-trajectory coverage in its frozen run but returned HOLD for every episode/task.

```text
coverage alone != useful stopping certificate
```

A retained set must be both sufficiently covered and sufficiently task-discriminative.

## Reduced repeated-cycle diagnostic

The ten-cycle study used a smaller configuration:

```text
TRAIN 80 / CALIBRATION 80 / TEST 60
max stage 16
points 80
```

It is **not** a repetition of the first full 160/160/120, K=20, 120-point final test. Its previously reported mean realized coverage of 91.17% characterizes only that reduced configuration and must not be used to reinterpret the first 85% final-test realization.

## Matched comparator study

`experiments/matched_baselines.py` was added after adversarial review to isolate whether the proposed certificate adds value beyond easier alternatives. Before opening its new comparison TEST seeds, the code freezes comparator parameters using CALIBRATION only.

The matched methods are:

```text
full budget
backend default estimator stop
risk-checked fixed stage
risk-checked estimator/residual threshold
risk-checked learned scalar task score
raw calibrated completion envelope
trajectory-conformal completion certificate
```

All share the same generated TRAIN/CALIBRATION/TEST episodes and task readers. Test outcomes are not used for tuning. This remains numerical evidence, not a real-backend result.

## Full formal reference

[`theory/FORMALIZATION_v1.md`](theory/FORMALIZATION_v1.md) gives one numbered reference (C1–C26)
for the whole mechanism: C1–C20 is exactly what this repository executes today; C21–C26 is a
proposed, unvalidated active-perception extension (choosing REFINE/NEW VIEW/RESET, not just
stopping) — read that document's status table before citing any part of it.

## Timing discipline

Numerical experiments generate complete trajectories and can sum stage costs to a counterfactual endpoint. These values are now named **trajectory-prefix costs**.

They do not establish speedup.

A real latency claim requires the university harness in

```text
timing_mode = online_policy_measured
```

with separately executed paired policies on declared hardware, including all policy-specific overhead.

## University lab test

A laboratory can connect any iterative 6D backend that exposes intermediate stages:

```bash
python lab/run_real_system.py \
  --config lab/config.example.json \
  --train train.jsonl \
  --calibration calibration.jsonl \
  --test test.jsonl \
  --out lab_results.json
```

See [`lab/README.md`](lab/README.md) and [`experiments/real_backend_protocol.md`](experiments/real_backend_protocol.md).

The lab harness reports:

```text
held-out trajectory coverage
certificate rate
k_C < k_E
certificate hitting time conditional on a hit
terminal HOLD endpoint
paired completion difference
finite-sample non-inferiority bound
HOLD
unsafe ACT
trajectory-prefix cost
measured online-policy latency only when supplied
```

## Automatic checks

GitHub Actions run on Ubuntu, macOS, and Windows. The workflow executes unit/adversarial tests, numerical experiments, matched comparator CI profile, university-lab smoke test, evidence-boundary assertions, and a larger Ubuntu evidence job.

Mechanical controls include:

- exact Kabsch and task-reader checks;
- no hidden-ground-truth parameter in online gate interfaces;
- disjoint TRAIN/CALIBRATION/TEST identifiers;
- augmented split-conformal small-n behavior;
- NaN/overflow fail-closed behavior;
- whole-trajectory calibration;
- `unsafe_act_on_covered_episode_count == 0` for the matching numerical readers;
- n=1 non-inferiority adversarial control;
- minimum sample-size enforcement;
- matched comparator parameters frozen before TEST;
- glosa claim-card validation.

## Evidence discipline

The repository applies the five-question / E-A-D discipline from **glosa — Rigour Without Infrastructure**:

```text
Q1 what was actually seen/run?
Q2 what does the record alone separate?
Q3 what did AI add?
Q4 what is assumed?
Q5 what independent check was run?
```

Current evidence ceiling remains **public numerical/mechanical evidence**. Code hardening and repeated CI do not manufacture independent human, real-sensor, or physical-robot validation.

## Claim boundary

Supported at the current numerical level:

- an iterative numerical 6D backend can construct a trajectory-calibrated completion set and use robust task-set inclusion as a stopping certificate;
- certificate correctness now fails closed at finite-sample and numerical boundary cases tested by adversarial regressions;
- high coverage can still be operationally useless when the set is too wide;
- the method can be compared reproducibly against matched calibration-tuned alternatives;
- a public backend-agnostic harness exists for real-system falsification;
- that harness has been exercised end-to-end on real RGB-D data (BOP LM-O) through a real (non-neural) ICP backend, producing a genuine falsifying result rather than a certificate (see "Real-sensor evidence" above) — this is in-house execution, not independent-lab validation, and does not by itself resolve any item in the OPEN/HOLD list below.

Still OPEN / HOLD:

- real RGB-D envelope coverage;
- real iterative neural-backend behavior;
- FoundationPose/GPU speedup;
- direct online-policy latency advantage;
- physical manipulation non-inferiority;
- physical safety certification;
- universal distribution-shift validity;
- universal novelty/superiority.

## Prior-art boundary

The repository does **not** claim to invent task-aware perception, early stopping, conformal prediction, or conformal 6D pose uncertainty.

The narrow candidate contribution is:

> **coverage-qualified downstream task stopping of iterative 6D pose refinement: stop when a calibrated pose-completion set has collapsed to one downstream task verdict, rather than waiting only for estimator-side convergence.**

## Author

Yaoharee Lahtee  
Open Civil Science Initiative, Bangkok, Thailand
