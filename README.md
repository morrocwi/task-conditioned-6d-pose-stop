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
10. **Third real-sensor ICP run, Bonferroni-corrected multi-checkpoint certificate (in-house, not independent)** — same backend/dataset/split/tolerances as run 9, changes only the certificate-construction method: a NEW code path (`lab/multicheckpoint.py`, `cqts/safety.py`'s Bonferroni functions) restricted to K'=4 predeclared checkpoint stages, calibrated independently at `alpha/K'` (Toledo proposal PROP-CONF-03; union bound machine-checked in `~/ANSE.ASIA/toledo/coq/canonical/PROP_CONF_03_union_bound.v`), instead of the existing joint whole-trajectory construction (unchanged, still used by runs 8-9). Refutes the specific falsifiable prediction it was designed to test: per-checkpoint quantiles came out LARGER, not smaller, than the joint quantile, and the certificate rate stayed at 0%. See "Real-sensor evidence" below.
11. **Fourth real-sensor ICP run, closed-form condition-number decay predictor (in-house, not independent)** — same backend/dataset/split/tolerances as runs 9-10, changes only the error-scale predictor: replaces C6's fitted log-linear model with a closed-form decay law (`lab/decay_predictor.py`, Toledo proposal PROP-DECAY-01) derived from the ICP normal-equations Hessian H_k's own condition number, combined with the (unchanged) whole-trajectory calibration construction. Refutes both the proposed H_k~graph-Laplacian analogy (structurally, confirmed numerically) and the predicted calibration improvement (the new predictor's calibrated q is LARGER, not smaller, than either prior construction). See "Real-sensor evidence" below.
12. **Fifth real-sensor cycle, native retained-sensitivity model, no ground truth anywhere online (in-house, not independent)** — a genuinely different decision mechanism (`lab/native_sensitivity.py`, Toledo proposals PROP-NATIVE-01/02): no fitted model, no calibrated completion set, no conformal quantile; ACT is licensed iff the task verdict is invariant under a bounded perturbation along H_k's own smallest eigenvalue direction. The first of five real-data cycles to license ACT at all — but ACT was WRONG (unsafe) on 92.5-100% of test episodes, always firing at the very first ICP stage. Reported as a serious safety finding, not a success. See "Real-sensor evidence" below.
13. **Sixth real-sensor cycle, memory/persistence accumulator (in-house, not independent)** — `lab/native_persistence.py`, Toledo proposal PROP-NATIVE-03: gates run5's unmodified single-instant check behind a resetting streak `M_k>=theta` (`theta=4`), plus a disclosed residual-scaled-magnitude variant. Refuted: the underlying single-instant check is near-constant across stages on both TRAIN and TEST, so the streak degenerates into a fixed-delay knob (ACT deterministically at `k=3` on 100% of test episodes, unsafe on 80.0-87.5% of them) rather than a genuine noise filter; the residual-scaled variant produced byte-identical outcomes because its dynamic range is far too small to matter. See "Real-sensor evidence" below.

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

### Third real-sensor cycle: Bonferroni-corrected multi-checkpoint certificate (PROP-CONF-03)

Run 2's own result pointed the diagnosis at the calibrated-envelope-CONSTRUCTION step rather than
task tolerances. A diagnosis recorded in glosa
(`~/ANSE.ASIA/glosa/projects/GLS-2026-005_pose-stop-conformal-diagnosis/DIAGNOSIS_HYPOTHESIS.md`)
identified the joint whole-trajectory max-over-15-stages nonconformity score (C7/PROP-CONF-02) as
the likely dominant driver of the ~7.4x envelope inflation (`q≈2.0008`, rank 37/40), and proposed a
Bonferroni-corrected fix: restrict the certificate to K'=4 predeclared checkpoint stages
(`{4,8,12,15}`, the largest feasible value at n=40 calibration episodes and alpha=0.1), each
calibrated independently at `alpha/K'=0.025` via the same `safe_split_conformal_quantile` used by
the existing construction, combined by a machine-checked union bound (Toledo proposal
PROP-CONF-03, `~/ANSE.ASIA/toledo/coq/canonical/PROP_CONF_03_union_bound.v`). This is a NEW code
path (`lab/multicheckpoint.py`; `lab/run_real_system.py --mode bonferroni_multicheckpoint`) — the
existing joint whole-trajectory construction used by runs 1-2 is unchanged and unaffected.

`lab/results/real-bop-lmo-2026-09-09-run3/` re-runs the identical dataset, split, ICP backend, and
run 2's TRAIN-derived tolerances, varying only the certificate-construction method. Predeclared
before FINAL TEST (`681c710`).

**Result: the tested falsifiable prediction is REFUTED.** The four per-checkpoint quantiles
(`q ∈ [2.085, 2.200]`) came out 4.3-10.0% LARGER than the joint whole-trajectory `q≈2.0008` used by
runs 1-2, not smaller as predicted, and the certificate rate stayed at exactly 0% for all three
tasks (100% HOLD), identical in kind to runs 1-2. The likely reason: splitting `alpha` four ways
pushes each checkpoint's required calibration rank to `40/40` — the single largest of only 40
calibration scores — and at this sample size that per-checkpoint Bonferroni penalty outweighs the
stage-aggregation penalty it was designed to remove. This does not show the union-bound
construction is incorrect (its own coverage guarantee, 97.5-100% all-checkpoints-covered against a
90% target, held); it refutes the more specific quantitative prediction that this restriction would
be numerically cheaper than the joint construction at n=40. Full accounting in
[`lab/results/real-bop-lmo-2026-09-09-run3/RESULT.md`](lab/results/real-bop-lmo-2026-09-09-run3/RESULT.md).

### Fourth real-sensor cycle: closed-form condition-number decay predictor (PROP-DECAY-01)

Founder authorization: use Toledo's real Coq-proven spectral ceilings (weld/M.40, weld/M.42) plus
the Mohar/Fiedler diameter floor `q_formal/M.07` (a bare citation, not Coq-proven) to replace C6's
fitted log-linear predictor with a decay law derived from the ICP normal-equations Hessian `H_k`'s
own condition number: `kappa_k=lambda_max(H_k)/lambda_2(H_k)`, `rho_k<=(kappa_k-1)/(kappa_k+1)`,
`s_k,i=rho_k^(K-k)*|residual_i|`. Toledo proposal `PROP-DECAY-01`
(`~/ANSE.ASIA/toledo/registry/proposals/spectral_decay_predictor.json`) disclosed up front that the
`H_k ~ L_R` identification is an ANALOGY, not a proven fact.

A companion attempt (in the Toledo repo, not this one) to Coq-prove `q_formal/M.07`'s diameter
floor `lambda_2 >= 4/(nD)` **did not close** in the time given — the classical Mohar 1991 proof
needs the full min-max/Courant-Fischer characterization of `lambda_2` over the whole space
orthogonal to the constant vector, a materially stronger apparatus than the single-Rayleigh-pair
style this workspace's existing spectral Coq files use, and was not completed. Independent of that,
exposing the real `H_k = J^T J` this backend computes (`lab/bop_icp_backend.py:normal_equations_H`,
a purely additive diagnostic; the Kabsch update itself is unchanged) shows the proposed analogy
**does not hold structurally here at all**: `H_k` is a fixed 6x6 SPD matrix over the 6 pose degrees
of freedom, with no vertex/edge structure and no dependence of its dimension on the correspondence
count — a graph "diameter D" is simply undefined for it. This run therefore uses the correct
quantity for `H_k` instead: the ordinary matrix condition number `kappa=lambda_max/lambda_min`
feeding the classical (cited) Gauss-Newton/Kantorovich contraction bound, which needs no graph
object and no `q_formal/M.07` at all.

`lab/results/real-bop-lmo-2026-09-09-run4/` reuses runs 9-10's identical dataset/split/tolerances
and the unmodified whole-trajectory (C7-C9) calibration construction (chosen over run3's
Bonferroni-checkpoint construction because run3 found it gives the smaller `q`), varying only the
predictor. Predeclared before FINAL TEST (`724c47a`).

**Result: the tested falsifiable prediction is REFUTED, on two independent fronts.** (1) The
proposed `H_k~L_R` analogy is refuted structurally and numerically (a real correspondence k-NN
graph built separately from the same ICP stage has n=400, diameter=16, giving `4/(nD)=0.000625` —
an unrelated number to `H_k`'s own `lambda_min=0.1235` at that stage). (2) The closed-form decay
predictor's calibrated envelope is not tighter than C6's: `q=3.298` (~27x inflation) is LARGER than
both the original joint `q≈2.0008` (runs 1-2) and every run3 Bonferroni per-checkpoint `q`
(2.085-2.200). Certificate rate remains exactly 0.0 for all three tasks (100% HOLD), the fourth
cycle in a row to find this; held-out coverage is 95.0% (Wilson 83.5-98.6%). Full accounting in
[`lab/results/real-bop-lmo-2026-09-09-run4/RESULT.md`](lab/results/real-bop-lmo-2026-09-09-run4/RESULT.md).

### Fifth real-sensor cycle: native retained-sensitivity model, no ground truth anywhere online (PROP-NATIVE-01/02, H3)

Founder authorization: after four calibration-side falsifications (runs 1-4, all `certificate_rate
= 0.0`), question the underlying error definition itself instead of the calibration construction.
Toledo proposals `PROP-NATIVE-01`/`PROP-NATIVE-02`
(`~/ANSE.ASIA/toledo/registry/proposals/native_retained_sensitivity.json`) replace the classical
pose-error definition `e_k := Log(That_k^-1 T*)` (a distance to an unobserved `T*`, a non-readout
under this workspace's own information-discrete-math discipline) with a retained-sensitivity ACT
rule that uses no ground truth anywhere online: perturb the current pose estimate `T_hat_k` along
the ICP normal-equations Hessian `H_k`'s own single smallest eigenvalue direction (magnitude
proportional to `1/lambda_min`, capped), and ACT iff the task reader's verdict is invariant across
the unperturbed pose and both perturbed candidates. This is a genuinely different mechanism from
runs 1-4 (`lab/native_sensitivity.py`): no fitted model, no calibrated completion set, no conformal
quantile. `lambda_ref`/`CAP` are derived from TRAIN data only
(`lab/derive_native_threshold_from_train.py`), frozen before FINAL TEST (`43954c09`).

**Result: this is the first of five real-data cycles to license ACT at all — and it is unsafe.** ACT
fired on 100% of test episodes for all three declared tasks, always at the very first ICP stage
(`k=0`, before any refinement iteration), and was WRONG (unsafe ACT) on 92.5-100% of those episodes
(114/120 task-episode pairs). Diagnosed cause: the perturbation magnitude this construction produces
is bounded well below the real coarse-detector initial-pose error, so the invariance test answers "is
a tiny wobble around the current estimate tolerable" rather than "is the current estimate close to
correct" — `H_k`'s spectral floor measures local geometric conditioning, not the absolute scale of
the estimate's own (unknown) residual error, and this run empirically refutes PROP-NATIVE-02's own
disclosed open assumption that the two are safely related. This is reported as a serious safety
finding, not a success — runs 1-4 all failed SAFE (100% HOLD); this is the first construction to
fail UNSAFE. Full accounting in
[`lab/results/real-bop-lmo-2026-09-09-run5/RESULT.md`](lab/results/real-bop-lmo-2026-09-09-run5/RESULT.md).

### Sixth real-sensor cycle: memory/persistence accumulator (PROP-NATIVE-03)

Founder authorization: run5's unsafe-ACT finding (PROP-NATIVE-02) motivated testing whether a
memory/persistence requirement recovers safety. Toledo proposal `PROP-NATIVE-03`
(`~/ANSE.ASIA/toledo/registry/proposals/native_retained_sensitivity.json`) wraps run5's unmodified
single-instant invariance check in a resetting streak accumulator `M_k` (`M_k=M_{k-1}+1` when the
check passes, `0` otherwise), gating `ACT <=> M_k >= theta` (`theta=4`, predeclared, reused from
run3's own frozen Bonferroni checkpoint set). Two variants were run: (a) theta-gate only; (b) also
scales the perturbation magnitude by the current observable ICP residual RMSE, addressing run5's
own diagnosed root cause directly.

**A TRAIN-only diagnostic, computed and disclosed before test.jsonl was opened, predicted the
outcome:** the single-instant check evaluates True at every one of 640 TRAIN stage readings for
all three tasks — there is no toggling for a streak counter to filter. **Result: as predicted, both
variants are refuted.** ACT fires deterministically at the fixed stage `k=theta-1=3` on 100% of test
episodes for all three tasks in both variants (byte-identical outcomes), unsafe on 80.0-87.5% of
them (99/120 task-episode pairs) — a modest drop from run5's 92.5-100%, attributable only to the 3
extra ICP iterations run before committing, not to any noise-filtering property of the persistence
mechanism. Variant (b)'s residual-scale multiplier reached only ~1.15-1.8x at a cap of 2x, roughly
two orders of magnitude too small relative to the real coarse-detector error scale to move the
result. Full accounting in
[`lab/results/real-bop-lmo-2026-09-09-run6/RESULT.md`](lab/results/real-bop-lmo-2026-09-09-run6/RESULT.md).

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
