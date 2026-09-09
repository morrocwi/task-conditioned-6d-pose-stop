# Formalization v1 — Completion/Imagination and Verification/Decision

Author: Yaoharee Lahtee, Open Civil Science Initiative. Recorded 2026-09-09.

This document is a single, numbered formal reference (C1–C26) for the mechanism this repository
implements, plus a clearly separated proposal for what it does not yet implement. It exists so a
reader never has to guess which equation is executable code and which is a target for future work.

**Non-claim, stated once, binding for the whole document:** the "Completion/Imagination" and
"Verification/Decision" naming is an architecture metaphor. It is not a neuroscience claim about
hemispheric lateralization or any other biological mechanism; real neuroscience is far more complex
than that metaphor and this document makes no claim about it.

## Relation to the other notes in this folder

`COVERAGE_TO_TASK_CERTIFICATE.md` states the C7–C9 statistical bridge in isolation;
`TOLEDO_COMPLETION_ENVELOPE.md` ties the completion-envelope construction to existing Toledo
equation-library anchors (e.g. `weld/M.01.v1`). This document is the single numbered reference for
the whole mechanism end to end (C1–C26) and is the one to update first; the other two remain valid
narrower notes and are not superseded, only subsumed for full-picture reading.

## Status at a glance

| Range | Status | Where it lives in this repo |
|---|---|---|
| C1–C20 | **Implemented and executable** | `cqts/safety.py`, `lab/run_real_system.py`, `experiments/*.py` |
| C20b | **Implemented and executable, a genuinely different decision path (no ground truth online), real BOP-LMO result: ACT-fires-unsafely** | `lab/native_sensitivity.py`, `lab/run_native_sensitivity.py`, `lab/results/real-bop-lmo-2026-09-09-run5/` |
| C21–C26 | **Proposed, not implemented, not validated** | none — future work |

Known, disclosed limitation shared by C1–C20b alike (`theory/CONTINUUM_AUDIT_20260909.md` item 1):
`T_hat_k`/`T*` are represented in `SE(3)`, a continuum manifold. C20b removes `T*` from the online
decision (its whole point) but does NOT fix that deeper continuum-representation injection — a
separate, unattempted design pass.

Do not describe C21–C26 as proved, tested, or existing in code anywhere in this repository's
README, CLAIMS.md, or any release material. If a future run implements and validates them, this
document is the place to update, with its own evidence, not a place to retroactively claim they
were already true.

---

## Part 0 — the readout the system actually has

Let the true pose be `T* ∈ SE(3)` and the estimator's pose at refinement stage `k` be `T̂_k ∈
SE(3)`. Define the local pose error through the Lie algebra:

```
e_k = Log(T̂_k^-1 T*) ∈ R^6,   e_k = [e_x, e_y, e_z, e_ωx, e_ωy, e_ωz]
```

The first three components are translation error; the last three are a local rotation-vector
error, not globally-valid Euler roll/pitch/yaw (see `lab/README.md`'s own unit note).

At deployment `T*` — and therefore `e_k` — is unknown. What the system actually has is an online,
ground-truth-free observable readout:

```
r_k = R(T̂_k, I, Δ_k, ρ_k, ...)
```

e.g. residual, proposed update magnitude, an ensemble/dispersion proxy, the stage index. This is
exactly `lab/ADAPTER_TEMPLATE.py`'s `observable_features(backend_stage)` contract: only
ground-truth-free quantities are allowed here.

---

## Part I — Completion / Imagination (implemented, C1–C10)

### C1 — the set of not-yet-excluded worlds

```
C_k = { z : z remains compatible with r_k }
```

The non-collapse this document insists on:

```
z ∈ C_k  =>  current evidence has not excluded z      (never: z ∈ C_k => z is true)

possible ≠ true
```

### C2 — task identity instead of pose identity

Let the downstream task reader be `O_T(z) ∈ {PASS, FAIL}`. Read every retained completion through
it:

```
Y_{T,k} = { O_T(z) : z ∈ C_k }
```

### C3 — the three-state verdict

```
V_{T,k} = PASS  if Y_{T,k} = {PASS}
          FAIL  if Y_{T,k} = {FAIL}
          HOLD  if Y_{T,k} = {PASS, FAIL}
```

Central distinction: **pose unresolved does not imply task unresolved.** A task that is
yaw-insensitive can be PASS under every retained completion even while yaw itself is still
unresolved.

### C4 — task ambiguity (imagination diameter)

For a task-outcome distance `d_T`:

```
W_T(C_k) = sup_{z,z' ∈ C_k} d_T(O_T(z), O_T(z'))
```

For a binary task, `d_T(PASS,PASS)=d_T(FAIL,FAIL)=0`, `d_T(PASS,FAIL)=1`, so `W_T(C_k) ∈ {0,1}`.
`W_T(C_k)=0` means every unresolved world is task-equivalent — not that pose has been fully
resolved.

### C5 — the calibration problem, named honestly

An uncalibrated `C_k` can be too narrow: if the true error is 5 cm but the system "imagines" only
±1 cm, every imagined world can pass while the real world sits outside the set entirely. This is
exactly the correctness defect the repo's own adversarial review found and fixed
(`evidence/ADVERSARIAL_REVIEW_RESPONSE.md`) — C6–C10 are the fix.

### C6 — an observable error-scale predictor (learned on TRAIN only)

```
s_{k,i} = f_{θ,i}(r_k) > 0

log s_{k,i} = β_i^T [1, log(p_{k,i}+ε_p), log(|Δ_{k,i}|+ε_Δ), log(ρ_k+ε_ρ), k/K]
```

`s_{k,i}` is a predicted error-scale shape, not a posterior: `s_{k,i} ≠ P(e_{k,i} | r_k)`.

### C7 — whole-trajectory conformal nonconformity score (CALIBRATION split only)

```
A_j = max_{k≤K} max_{i≤6} [ log(|e_{j,k,i}| + δ_i) − log(s_{j,k,i}) ]
```

Maxing across the whole trajectory (not per-stage) is deliberate: the stopping stage is chosen
adaptively later, so per-stage independence cannot be assumed.

### C8 — the corrected split-conformal quantile (post-review)

```
r_α = ceil((n+1)(1-α))

q_α = A_(r_α)          if r_α ≤ n
q_α = +∞                if r_α = n+1
```

**The binding correction from the adversarial review**: `r_α` is never clamped back to `n`. When
calibration data is too small for the requested level, `q_α = +∞`, which forces HOLD rather than
issuing a falsely-narrow certificate. Implemented exactly as
`cqts/safety.py::safe_split_conformal_quantile`; the fail-closed HOLD path is
`cqts/safety.py::fail_closed_task_pass` together with `CertificateNumericsError`.

### C9 — the calibrated envelope half-width

```
b_{k,i} = exp(log s_{k,i} + q_α) − δ_i
```

Implemented as `cqts/safety.py::safe_exp_error_bound`.

### C10 — the calibrated completion envelope

```
Ĉ^cal_k = { e ∈ R^6 : |e_i| ≤ b_{k,i} for all i }
```

This is the executable form of "the set of pose errors that current evidence plus calibration
still permits."

### C9b — Bonferroni-corrected multi-checkpoint alternative to C7 (implemented, extends C7–C9)

An extension of the C7–C9 mechanism above, not part of the C21–C26 active-perception proposal
below: instead of one joint nonconformity score maxed over all K stages AND 6 coordinates (C7),
restrict certificate checks to K' predeclared checkpoint stages `k_1,...,k_{K'}` and calibrate each
independently:

```
A_{j,k_m} = max_{i≤6} [ log(|e_{j,k_m,i}| + δ_i) − log(s_{j,k_m,i}) ]         (coordinate-max only)

q_{k_m} = SplitConformalQuantile({A_{j,k_m}}_{j=1}^n, α/K')                    (same C8 order statistic)
```

then build the envelope at each checkpoint exactly as C9 (`b_{k_m,i} = exp(log s_{k_m,i} + q_{k_m}) − δ_i`).
By the union bound (Boole's inequality), `P(any of the K' checkpoints miscovers) ≤ Σ α/K' = α`, so
the combined per-checkpoint constructions give the same `≥ 1−α` whole-trajectory coverage guarantee
as the joint C7 construction, without its 15-way stage-aggregation term. Feasibility requires
`ceil((n+1)(1-α/K')) ≤ n`; for this repository's real-data runs (n=40, α=0.1) this gives K'≤4
feasible, K'=5 already forcing `q=+∞` at every checkpoint.

Registered in Toledo as proposal **PROP-CONF-03**
(`~/ANSE.ASIA/toledo/registry/proposals/conformal_stopping_family.json`; C7/C8 above correspond to
retroactively-registered **PROP-CONF-01/02**). The union-bound combination step is machine-checked,
axiom-free, over ℚ: `~/ANSE.ASIA/toledo/coq/canonical/PROP_CONF_03_union_bound.v`
(`finite_union_bound`, `bonferroni_checkpoints`, both `Print Assumptions` = "Closed under the
global context"). The per-checkpoint split-conformal exchangeability guarantee it composes (C8 at
level α/K') is cited from standard conformal-prediction theory, not itself re-derived here.

Implemented as a NEW code path (`lab/multicheckpoint.py`,
`cqts/safety.py::safe_multi_checkpoint_quantiles` / `bonferroni_feasible_max_checkpoints` /
`bonferroni_checkpoint_alpha`), additive alongside the unmodified joint C7–C9 construction so
existing results stay reproducible (`lab/run_real_system.py --mode whole_trajectory` vs.
`--mode bonferroni_multicheckpoint`). Executed once, on real BOP LM-O data, in
`lab/results/real-bop-lmo-2026-09-09-run3/` (diagnosis: `~/ANSE.ASIA/glosa/projects/GLS-2026-005_pose-stop-conformal-diagnosis/DIAGNOSIS_HYPOTHESIS.md`).
Result: **the falsifiable prediction that this construction's `q_{k_m}` would be smaller than the
joint C7 construction's `q≈2.0008` was refuted** — the four checkpoints' quantiles (`q ∈ [2.085,
2.200]`) came out 4.3–10.0% larger, because at n=40 calibration episodes every checkpoint's
Bonferroni-corrected rank lands at `40/40` (the single largest calibration score), and that
per-checkpoint penalty outweighed the removed stage-aggregation term. The construction's own
coverage guarantee held (all-checkpoints-covered rate 97.5% against a 90% target); what is refuted
is the specific quantitative prediction that it would be cheaper than C7 at this sample size, not
the union-bound argument itself. See `lab/results/real-bop-lmo-2026-09-09-run3/RESULT.md` for full
accounting.

### C6b — closed-form condition-number decay predictor, alternative to C6 (implemented)

An alternative to C6 above (not to C7–C9, which are reused unchanged): instead of fitting `β_i` by
TRAIN regression, derive `s_{k,i}` in closed form from the ICP normal-equations Hessian
`H_k = J^T J` (the same Jacobian `lab/bop_icp_backend.py::local_proxy` already builds internally,
now also exposed directly via `normal_equations_H`):

```
kappa_k = lambda_max(H_k) / lambda_min(H_k)                    (ordinary matrix condition number)
rho_k   = (kappa_k - 1) / (kappa_k + 1)                        (Kantorovich/Gauss-Newton contraction bound, cited)
s_{k,i} = rho_k^{K-k} * max(|proxy_{k,i}|, delta_i)            (proxy = C6's own local-dispersion feature, un-logged)
```

Toledo proposal **PROP-DECAY-01**
(`~/ANSE.ASIA/toledo/registry/proposals/spectral_decay_predictor.json`) originally proposed reading
`H_k` as an instance of the graph-Laplacian family `L_R := D_W - W`, so that `q_formal/M.07`'s
diameter-based Fiedler floor `lambda_2 >= 4/(nD)` could lower-bound `H_k`'s conditioning (using
`lambda_2` in the graph-Fiedler sense — 2nd smallest of an n-node Laplacian's eigenvalues, not the
ordinary 2nd-largest of a small dense matrix's spectrum). Two independent checks refuted this:

1. A companion Coq-proof attempt on `q_formal/M.07`'s bound itself (paired Toledo-repo task, not
   this repository) did not close — Mohar's 1991 proof needs the full min-max/Courant-Fischer
   characterization of `lambda_2` over the whole space orthogonal to the constant vector, stronger
   than the single-Rayleigh-pair style this workspace's existing spectral Coq files
   (`InfoSpectralCeilingSharp.v`, `RDL_SpectralCeiling.v`) use.
2. Exposing the real `H_k` this backend computes shows the `H_k ~ L_R` identification does not hold
   structurally regardless: `H_k` is a FIXED 6x6 SPD matrix (one row/column per pose degree of
   freedom), with no vertex/edge structure and no dependence of its size on the correspondence
   count `n_k` (more inliers change its six eigenvalues quantitatively, never its shape) — a graph
   "diameter D" is undefined for it. `kappa_k`/`rho_k` above therefore use the ORDINARY matrix
   condition number (its smallest of only 6 eigenvalues, not a Fiedler value), which is exactly
   what the classical (cited) Gauss-Newton contraction bound needs — no graph object, and no
   `q_formal/M.07`, are required at all.

Implemented as a NEW code path: `lab/decay_predictor.py` (`kappa_rho_from_H`,
`decay_predicted_log_error`, `DecayModel`), wired into `lab/run_real_system.py::predicted_log_error`
via a documented, additive `isinstance(model, DecayModel)` branch — C6's own fitted-regression
branch is untouched for any non-`DecayModel` caller. `lab/bop_icp_backend.py::normal_equations_H`
exposes `H_k` as a purely additive diagnostic field (`ICPStage.hessian`); the Kabsch closed-form
update itself is unchanged. Executed once, on real BOP LM-O data, combined with the unmodified
whole-trajectory (C7–C9) calibration construction — chosen over C9b's Bonferroni-checkpoint
construction because C9b's own run (run3) found it gives a smaller `q` — in
`lab/results/real-bop-lmo-2026-09-09-run4/`.

**Result: refuted.** The calibrated `q=3.298` (~27x multiplicative envelope inflation) is LARGER
than both C6's original joint `q≈2.0008` (runs 1–2) and every C9b per-checkpoint `q` (2.085–2.200);
certificate rate stays at exactly 0.0 for all three declared tasks (100% HOLD), the fourth cycle in
a row to find this. A real correspondence k-NN graph built separately (diagnostic only, not part of
the predictor) from one ICP stage confirms the structural mismatch numerically: `n=400`,
`diameter=16`, `4/(nD)=0.000625`, an unrelated number to that same stage's `H_k` `lambda_min=0.1235`.
Disclosed, not-yet-independently-re-verified diagnosis for why the closed-form predictor
underperforms: measured `rho_k` across TRAIN stages clusters tightly near 1 (median 0.9991), likely
because `H_k`'s rotation columns (`-skew(R p_i)`, scaled by point-coordinate magnitude) and
translation columns (identity-scaled) mix units, so `kappa_k` is dominated by this
parameterization-scaling artifact rather than by the correspondence set's true geometric
conditioning. See `lab/results/real-bop-lmo-2026-09-09-run4/RESULT.md` for full accounting.

---

## Part II — Verification / Decision (implemented, C11–C20)

### C11 — task-admissible region

```
A_T = { e : g_T(e) ≤ 0 }
```

e.g. `label_alignment`: `A = { e : |e_i| ≤ τ_i, ∀i }`. `keyed_insertion` needs a joint constraint,
not independent per-axis thresholds, because small per-axis errors can still combine into a part
that will not seat:

```
|e_x|/τ_x + |e_y|/τ_y + |e_ωz|/τ_ω ≤ 1
```

Implemented via `cqts/safety.py::validate_task_spec` / `fail_closed_task_pass` and the task
definitions in `lab/config.example.json`.

### C12 — the coverage-qualified certificate

```
ACT is licensed  <=>  Ĉ^cal_k ⊆ A_T   <=>   O_T(e) = PASS  for all e ∈ Ĉ^cal_k
```

### C13 — certificate hitting time

```
k_C = inf{ k : Ĉ^cal_k ⊆ A_T },   k_C = +∞ if no stage satisfies it
```

A terminal HOLD at budget `K` must never be relabelled as `k_C` — this is the exact corrected
semantics from `README.md`'s "What changed after adversarial review."

### C14 — the decision rule (this repository's current, shipped policy)

```
D_{T,k} = ACT       if Ĉ^cal_k ⊆ A_T
          CONTINUE  if k < K and no certificate yet
          HOLD      if k = K and no certificate yet
```

### C15 — connecting to estimator-side stopping

```
k_E = inf{ k : C_E(r_k) = STOP }        (the backend's own stopping rule)

the event of interest:  k_C < k_E
```

`k_C < k_E` means the downstream task resolved before the estimator itself would call convergence
— the computational opportunity this repository exists to test.

### C16 — marginal coverage guarantee

Under split-conformal exchangeability:

```
P( e*_k ∈ Ĉ^cal_k  for all k ≤ K ) ≥ 1 − α
```

This is **marginal coverage**, not `P(robot safe) ≥ 1-α` — do not conflate the two in any writeup.

### C17 — from coverage to task correctness, by set inclusion (not confidence)

If `e*_k ∈ Ĉ^cal_k` (coverage holds this episode) and the system ACTs because `Ĉ^cal_k ⊆ A_T`,
then by plain set inclusion `e*_k ∈ A_T`, i.e.:

```
ACT ∧ coverage  =>  PASS_T
```

This is set-theoretic reasoning, not a confidence score — the repo's central methodological claim.

### C18 — non-inferiority against estimator-side stopping (post-review statistics)

The repo does not use a percentile bootstrap for this comparison (that was a real defect the
adversarial review found). It uses paired binary discordance with a conservative, finite-sample
Clopper–Pearson lower bound, requiring a predeclared margin, confidence level, minimum N, and
sampling unit — implemented in `cqts/safety.py::paired_binary_noninferiority`
(`clopper_pearson_lower`/`_upper`, `_binom_cdf`, `_binom_sf_ge`).

### C19 — timing claims are structurally separated

`trajectory_prefix_estimate` (descriptive/debugging only) is never sufficient for a latency claim.
A latency PASS/FAIL requires `timing_mode = online_policy_measured`: two full, separately executed
policies with feature/gate/update/synchronization overhead included — see `CONTRIBUTING.md`
"Timing claims" and `experiments/real_backend_protocol.md`.

### C20 — what is empirically demonstrated today (numerical, matched baselines)

The matched-baseline numerical study already shows the qualitative shape of the claim: on
`keyed_insertion`, the certificate matches estimator-default completion (95%) while moving unsafe-
ACT mass (5% → 0% observed) into HOLD (0% → 5%), at a lower mean endpoint (13.100 → 11.025). This
is real, already-executed evidence (`experiments/matched_baselines.py`,
`lab/results/*`), on numerical/synthetic and now one real-sensor (BOP-LMO) run — see
`lab/results/real-bop-lmo-2026-09-09/RESULT.md` for the real-sensor outcome (a falsifying result at
that run's specific tolerances/budget, reported honestly, not folded into this section as a
success).

### C20b — native retained-sensitivity ACT rule, an alternative to C7–C14 (implemented, no ground truth online, PROP-NATIVE-01/02)

Placement note: this belongs here, adjacent to the implemented C1–C20 family, and NOT in Part III
below — it is executed code with a real result on real data (`lab/native_sensitivity.py`,
`lab/run_native_sensitivity.py`, `lab/results/real-bop-lmo-2026-09-09-run5/`), unlike C21–C26's
proposed-only active-perception extension. It is a genuinely DIFFERENT decision path from C7–C14
above, not an addition to them: it replaces the fitted error-shape model (C6/C6b), the calibrated
completion envelope (C7–C10), and the coverage-qualified certificate (C11–C14) all at once, with a
single ground-truth-free invariance test. Runs 1–4's own C7–C14 code paths are untouched and remain
independently reproducible; this is a parallel path, per `ops/HANDOFF_2026-09-09_external_dataset.md`
("Fifth cycle authorized, 2026-09-09: H3").

**Motivation (PROP-NATIVE-01,
`~/ANSE.ASIA/toledo/registry/proposals/native_retained_sensitivity.json`):** every construction
above (C7–C14) is built around comparing an estimate against a (frozen, calibrated) notion of
distance to the unobserved true pose `T*` — even C7–C9's split-conformal calibration only ever sees
`T*` on the CALIBRATION split, never online, but it is still an ingredient of the calibration itself.
`e_k := Log(That_k^{-1} T*)` is a distance to a non-readout under this workspace's own
information-discrete-math discipline. PROP-NATIVE-01 proposes a retained-difference replacement that
needs no `T*` anywhere, not even offline for calibration: `Delta_k := delta_R(That_k, That_{k-1})`,
the directly observed update between consecutive pose readouts (already computable — this is exactly
`stage.delta_t_norm`/`stage.delta_r_norm`, already an existing observable feature).

**Decision rule (PROP-NATIVE-02):** at stage `k`, take the real ICP normal-equations Hessian
`H_k = J^T J` (C6b's own diagnostic, `lab/bop_icp_backend.py::normal_equations_H`, reused unchanged)
and its full eigendecomposition (`lab/decay_predictor.py::eigh_H`, additive to C6b's
`kappa_rho_from_H`, same symmetrization/`eigh` routine family). Select the single smallest
eigenvalue/eigenvector `(lambda_min, v_min)` — the direction the correspondence data itself
constrains least. Perturb `T_hat_k` by `+-m*v_min`, `m = min(C/lambda_min, CAP)` (`C := CAP *
lambda_ref`; `lambda_ref`, `CAP` both derived from TRAIN data only,
`lab/derive_native_threshold_from_train.py`):

```
ACT_k  iff  fail_closed_task_pass(spec, 0) == fail_closed_task_pass(spec, |dev(T_hat_k, T_hat_k (+) m v_min)|)
                                          == fail_closed_task_pass(spec, |dev(T_hat_k, T_hat_k (+) (-m) v_min)|)
```

where `dev` is `lab/se3.py::abs_pose_error_6d` (reused unchanged, the SAME convention C17/oracle
scoring uses) applied to the SE(3) deviation between `T_hat_k` and each perturbed candidate, and
`fail_closed_task_pass` is `cqts/safety.py`'s reader, reused unchanged. Since the unperturbed case is
`T_hat_k` compared to itself (zero deviation), it trivially reads PASS for any task spec with
strictly positive tolerances — this is disclosed explicitly, not hidden: the rule reduces exactly to
"ACT iff both perturbed candidates also PASS." **No ground truth appears anywhere in this rule.**
Oracle `abs_pose_error_6d` is read only afterward, by the evaluator, in exactly the C17-style
offline-oracle role — checking whether ACT-licensed episodes were actually correct and CONTINUE/HOLD
episodes were honestly uncertain — never as a calibration or decision input.

**Honest floor caveat (this is the load-bearing limitation of the whole construction):** unlike C6b's
refuted `H_k ~ L_R` (graph-Laplacian) identification, this construction does not need any graph
structure or diameter for `H_k` at all — it only uses `H_k`'s own eigenvalues directly, which exist
for any real symmetric matrix (PROP-NATIVE-02's stated advantage over PROP-DECAY-01). But it still
has NO proven floor for "how small is too small an eigenvalue": `q_formal/M.07`'s diameter-based
floor `lambda_2 >= 4/(nD)` was already shown (C6b, run4) not to apply to `H_k`'s own structure
(no vertex/edge structure, no diameter), and remains unmechanized regardless (its own Coq attempt did
not close — `~/ANSE.ASIA/toledo`). `CAP` here is therefore a declared/relative, TRAIN-derived bound
(p75 of the existing local-dispersion-proxy norm), **not a bound resting on any proven floor**. This
matters doubly because of `~/ANSE.ASIA/toledo/docs/NAVIER_STOKES_THROUGH_OUR_LENS.md`'s kappa-scaling
honesty note (and `docs/L_R_SPECTRAL_CEILING_AND_FLOOR.md`'s own ceiling/floor pair): that note shows
a condition-number-style bound `kappa <= (ceiling)/(floor)` is *not* uniform as a correspondence
graph's size/diameter grows — it grows with graph size, by construction of the two proven/cited
bounds themselves, not by assumption. This construction leans on the same family of spectral
quantities (`H_k`'s own eigenvalues) without yet having an analogous proven scaling law for
`lambda_min(H_k)` itself; the declared/relative `CAP` used here should be read as a placeholder for
a real bound, not as one.

**Result (`lab/results/real-bop-lmo-2026-09-09-run5/RESULT.md`): this is the first of five real-data
cycles (this run and C6/C6b/C9b's runs 1–4) to license ACT at all, and it does so unsafely.** ACT
fired on 100% of test episodes for all three declared tasks, always at `k=0` (the very first ICP
stage, before any refinement), and was wrong (unsafe ACT) on 92.5–100% of those episodes (114/120
task-episode pairs). Diagnosed cause: the perturbation magnitude this rule produces is bounded well
below the real coarse-detector initial-pose error (`lab/bop_icp_backend.py::perturb_pose`: 15mm
translation sigma, 5–20deg rotation), so the invariance test answers "is a tiny wobble around the
CURRENT estimate tolerable" rather than "is the current estimate close to correct" — `H_k`'s spectral
floor measures local geometric conditioning of the correspondence set, not the absolute scale of the
estimate's own (unknown) residual error relative to ground truth. This is the exact empirical check
PROP-NATIVE-02's own `honest_caveats` called for ("the relationship between 'small eigenvalue
direction' and 'actual pose error along that direction' is itself an assumption ... must be checked
empirically") — **and this run refutes that assumption**, at least at this trajectory stage, on this
backend. Reported as a serious safety finding: runs 1–4 all failed SAFE (100% HOLD, never an unsafe
ACT); this is the first construction in this line to fail UNSAFE instead.

**What this does and does not establish:** does NOT establish that H3/PROP-NATIVE-01/02 is a viable
stopping certificate on this backend, nor that PROP-NATIVE-01's broader native-error critique is
wrong (only PROP-NATIVE-02's specific spectral-floor-only perturbation-magnitude construction was
tested). DOES establish that `H_k`'s spectral floor alone, without pairing to an absolute
residual/noise-scale observable (the way C6b's own decay predictor pairs `rho_k` with the observable
residual proxy `|proxy_i|`), is not a safe proxy for pose-estimate uncertainty on this backend — a
concrete, transferable lesson for any future construction reusing `H_k`'s eigenstructure.

---

## Part III — Proposed extension: active perception (NOT implemented, C21–C26)

Everything below is a formal proposal for a future project phase. The objective functions are
written down; the transition/action model they depend on is **not validated**, and no code in this
repository implements C21–C26. Do not cite this section as evidence of anything beyond "the target
exists on paper."

### C21 — worst-case ambiguity after a perception action

For a perception action `a` in a candidate action set `U = {REFINE_x, REFINE_y, REFINE_ω, NEW_VIEW,
MOVE_CAMERA, RESET, ...}`, let `Φ_a(C_k)` be the set of possible completion sets after taking `a`:

```
W̄_T(a | C_k) = sup_{C' ∈ Φ_a(C_k)} W_T(C')
```

### C22 — ambiguity contraction

```
G_T(a | C_k) = W_T(C_k) − W̄_T(a | C_k)
```

### C23 — task-relevant information value per unit cost

For a perception-action cost `c(a) > 0`:

```
J_T(a | C_k) = G_T(a | C_k) / c(a)
```

### C24 — the proposed action-selection rule

```
a*_k = argmax_{a ∈ U} J_T(a | C_k)
```

In plain language: if the system cannot yet ACT, do not refine indiscriminately — pick the
perception action that removes the most task-relevant ignorance per unit cost.

### C25 — the proposed full policy

```
π(r_k) = ACT     if Ĉ^cal_k ⊆ A_T
         a*_k    if W_T(Ĉ^cal_k) > 0 and a worthwhile action exists
         HOLD    otherwise
```

### C26 — the proposed training objective

Not minimizing pose error alone (`min ‖e_k‖`), but a declared industrial trade-off:

```
min_π  E[ C_perception(π) + λ_F L_task + λ_U I_unsafe + λ_H I_HOLD ]
```

where the `λ` weights are a declared operating point (e.g. `λ_U ≫ λ_H` when an unsafe action is far
costlier than one abstention) — HOLD is then an optimal industrial decision under a declared cost
structure, not a default failure mode.

---

## The single sentence this whole document reduces to

```
Complete state knowledge ≠ sufficient knowledge for action.

Do not resolve all uncertainty; resolve only the uncertainty that can still change the task outcome.
```

Architecture shape:

```
Observe -> Constrained Imagination -> Calibration -> Task Equivalence -> {ACT, REFINE, NEW VIEW, RESET, HOLD}
```

C1–C20 are the executed half of this diagram (Observe → Constrained Imagination → Calibration →
Task Equivalence → {ACT, CONTINUE, HOLD}). C21–C26 (REFINE/NEW VIEW/RESET as a chosen, not fixed,
action) are the proposed, unvalidated extension toward a general decision architecture under
incomplete knowledge, for which iterative 6D pose refinement is the first candidate executable and
falsifiable domain — not the final scope.
