# Handoff — external real-dataset validation for task-conditioned-6d-pose-stop

Founder request (verbatim): after relaying a status report from a prior 26-minute agent run that
hardened this repo post-adversarial-review (v0.6, matched baselines, CI green, external-lab
protocol published in CONTRIBUTING.md/lab/), the founder said: "ไปหา ดาต้าเซตภายนอกมารันให้หน่อยและ
ยกระดับเรโป" — go find an external dataset, run it, and level up the repo. The repo's own
README/RESEARCH_QUESTION.md/evidence/ADVERSARIAL_REVIEW_RESPONSE.md are explicit that the current
evidence ceiling is numerical/synthetic only (numerical_icp.py); the acknowledged bottleneck is
real RGB-D -> real iterative pose backend -> calibration -> k_C/k_E -> direct online timing.

Repo root: task-conditioned-6d-pose-stop (public,
github.com/morrocwi/task-conditioned-6d-pose-stop, mirrored to local Forgejo). This repo has its
own strict no-overclaim culture (see evidence/ADVERSARIAL_REVIEW_RESPONSE.md, CLAIMS.md,
CONTRIBUTING.md) — never relax it for this task.

## What already exists (read before writing anything)
- `lab/README.md`, `lab/ADAPTER_TEMPLATE.py`, `lab/episode.schema.json`, `lab/run_real_system.py`,
  `lab/se3.py`, `lab/config.example.json`, `lab/make_example_data.py`, `lab/RESULT_TEMPLATE.md`,
  `lab/EXTERNAL_VALIDATION_CALL.md` — the exact interface a real backend must produce.
- `CONTRIBUTING.md` — minimum contribution package + predeclaration + TRAIN/CAL/TEST discipline.
- `RESEARCH_QUESTION.md` — H1-H4 real-backend hypotheses this run can speak to.
- `cqts/safety.py` — the shared correctness core (fail-closed numerics, conformal quantile,
  paired non-inferiority) — reuse it, never reimplement.
- `evidence/ADVERSARIAL_REVIEW_RESPONSE.md`, `CLAIMS.md` — what is and is not claimed today.

## Plan
1. Environment check done: scipy present, no open3d/cv2/trimesh; network to
   bop.felk.cvut.cz confirmed reachable; 171GB free disk.
2. Download a SMALL real subset of a standard public RGB-D 6D-pose benchmark (BOP format
   recommended — e.g. LineMOD (LM) or LM-O test set: RGB, depth, per-frame camera intrinsics
   `scene_camera.json`, ground-truth poses `scene_gt.json`, and the object's own textured/geometric
   mesh under `models/`). Scope to 1-2 objects and a bounded number of frames (documented, not
   silently truncated) to keep this tractable.
3. Build a REAL iterative pose backend from scratch with numpy+scipy only (scipy.spatial.cKDTree
   for correspondences, SVD/Kabsch for the rigid update per iteration) doing point-to-point or
   point-to-plane ICP between the depth-derived point cloud and the object mesh, seeded from a
   documented, honestly-labelled perturbation of ground truth (BOP raw data has no built-in coarse
   detector; state this plainly as a simulated coarse initializer, never as a full autonomous
   detection pipeline).
4. Wire it through `lab/ADAPTER_TEMPLATE.py`'s exact functions (observable_features,
   estimator_stop, incremental_ms, abs_pose_error_6d) -> `export_episode` -> JSONL matching
   `lab/episode.schema.json` exactly (validate against it).
5. Follow the lab protocol exactly: predeclare everything CONTRIBUTING.md/lab/README.md ask for
   BEFORE looking at final-test results, keep TRAIN/CALIBRATION/FINAL TEST disjoint and frozen,
   run `lab/run_real_system.py` (reuse it, don't fork its logic).
6. Produce the CONTRIBUTING.md minimum package (lab_results.json, backend_and_hardware.md,
   config.json, split_manifest.json, exact commit) under a new `lab/results/<run-id>/` directory.
7. Write up findings honestly against H1-H4 — this is REAL sensor data + a REAL iterative backend,
   which is a genuine step up from the numerical fixture, but it is NOT independent external-lab
   validation (still produced in-house) — label it exactly that way, update CLAIMS.md/README only
   with what the evidence supports, and do not remove or soften any existing HOLD/evidence-ceiling
   language unless this run genuinely retires it.
8. Full test suite must still pass; leak-scan; no AI vendor names; plain commit messages (check
   for any active session attribution instruction first). Push to origin AND local (Forgejo).

## If genuinely blocked
Report exactly what blocked it (download too large/slow, missing ground-truth format, no usable
object mesh, correspondence step degenerate, etc.) rather than fabricating a result. A partial,
honestly-reported attempt is worth more here than a fabricated PASS — this repo's own culture
already survived one adversarial review by refusing to do that.

## Novelty-claim constraint (founder relayed, 2026-09-09, from a literature-check pass)

Literature check through 2026 found: conformal 6D-pose uncertainty sets (Yang & Pavone 2023; Wang
et al. ICCV 2025), task-aware/task-conditioned perception (established 2026 robotics work),
conformal-guided early stopping in general ML (established), and even 2026 robotics work using
conformal calibration to gate manipulation-policy rollout termination (established) are ALL
already prior art. Do NOT claim any of those individually as novel, and NEVER use the word
"first" — a literature search alone cannot establish universal priority.

The one candidate the founder approved for the paper, exact wording, do not paraphrase looser:

> "We study a specific composition not established by the prior work reviewed here: using a
> trajectory-calibrated 6D pose-completion set itself as a downstream task stopping certificate
> for iterative pose refinement, with ACT licensed only when the entire retained set lies inside
> the declared task-admissible region."

If this run's results/README/CLAIMS.md touch a novelty statement, use this sentence (or a strict
subset of it), never a broader claim. The matched-baseline framing already in the repo (scalar
threshold can stop earlier while still producing unsafe ACT; the certificate trades a small HOLD
rate for lower unsafe-ACT) is exactly the kind of narrow, evidence-backed distinction the founder
wants kept, not "we are faster."

## Release plan (founder, verbatim): "ultracode ทำงานจากด้าต้าเซตภายนอกพร้อมสรุปผลได้เลยและออกรีลีส"
= ultracode the external-dataset work through to a summary and a release. Sequencing chosen to
avoid a file race: the single background Agent (already dispatched, see above) is still running
the actual download+ICP+lab-protocol work on this same repo's files. Do NOT launch a second agent
touching the same files concurrently. As soon as that agent reports done, launch an ultracode
Workflow (this repo's own culture demands independent review, never self-certification) for:
(1) independent adversarial review of the produced lab_results.json/config/writeup against
CONTRIBUTING.md's own checklist and the honesty guardrails above (a materially separate pass, not
a restatement); (2) a clean human-readable results summary (matching the founder's own report
table format: policy / mean endpoint / completion / HOLD / unsafe ACT, or an honest "did not
complete X" if that's what happened); (3) only if the review passes (or founder accepts disclosed
findings), tag+push+GitHub release on both origin and local — never release over an unresolved
BLOCK. If the review finds the run did not actually succeed (blocked, degenerate, or overclaiming),
report that plainly instead of releasing anyway.

## Positioning constraint (founder relayed, 2026-09-09, reviewer-simulation pass)

Hard rule for any writeup this or a later run produces: NEVER write "we solve 6D pose estimation"
or anything equivalent. The approved framing, precise, do not loosen it:

> "We are not trying to eliminate perception error entirely. We ask whether the error that remains
> still affects what the robot is about to do."

Explicitly out of scope and never claimed as done by this project: learning rotation on SO(3),
building a new pose estimator, data efficiency / viewpoint generalization, multi-object detection.
This project only does: reducing pose error's effect on the downstream robot task, declaring when
pose is "good enough for this task," using calibrated (not raw) uncertainty, and abstaining (HOLD)
instead of risking an unsafe ACT.

Evidence-honesty note relevant to the current run: classical ICP/Kabsch refinement against a real
BOP-format point cloud (this run's own contribution) is a genuine step past a purely synthetic
generator, but it is still NOT a learned/neural iterative pose backend (e.g. FoundationPose-class).
State this distinction explicitly in any results writeup — do not blur "real sensor data" with
"real learned vision backend." If a future run adds a real learned backend and k_C<k_E, T_C<T_E,
and S_C>=S_E-delta all hold together with the unsafe-ACT/HOLD story, that is the founder's stated
threshold for a strong ICRA/IROS/RA-L-shaped contribution (not necessarily CVPR, which wants a
core vision/estimation contribution this project explicitly does not make).

## Result received, review launched
Real BOP-LMO run done (commit fd65d27, pushed to local only): genuine falsifying result — 0%
certificate rate on all 3 tasks (calibrated envelope too wide for the reused tolerances), coverage
87.5% (CI straddles 90% target). Full detail lab/results/real-bop-lmo-2026-09-09/RESULT.md.
Ultracode review+release workflow launched: wf_934b7d58-e64. It will NOT push to origin or create
a GitHub release itself — reads its report, then the chair does the actual public tag/push if
ops/RELEASE_NOTES_v0.7_DRAFT.md says RELEASE READY: yes.

## Roadmap offered by founder, 2026-09-09 — SEE -> IMAGINE -> CALIBRATE -> CHECK -> DECIDE -> ACT/LOOK-AGAIN

Expands the two-faculty framing above into a 6-stage target architecture. Honest status against it
(repo state as of commit c84ee93 + the uncommitted-to-origin real-data run fd65d27):

- SEE (adapter ingesting a real backend's per-stage state): EXISTS as a spec (`lab/ADAPTER_TEMPLATE.py`)
  and was exercised for the first time this session against a real ICP backend (not yet FoundationPose
  or another learned backend).
- IMAGINE (completion set C_k of not-yet-ruled-out states): EXISTS, this is the repo's core mechanism.
- CALIBRATE (statistical coverage via `cqts/safety.py`'s conformal quantile, fail-closed on
  NaN/overflow/degenerate-n): EXISTS, hardened after the adversarial review.
- CHECK (task reader over every retained completion): EXISTS (`Y_T,k` / `fail_closed_task_pass`).
- DECIDE beyond ACT/CONTINUE/HOLD (choosing among REFINE/NEW_VIEW/MOVE_CAMERA/RESET via an
  information-value objective `a* = argmax_a G_T(a|C_k)/c(a)`): DOES NOT EXIST YET. The founder's own
  words: "สมการ G_T,J_T มีอยู่ใน theory แล้ว แต่ transition/action model ยังต้อง validate จริง" (the
  objective exists on paper, the transition/action model is unvalidated) -- this is real, unbuilt,
  unvalidated future work, not a documentation gap. Do not claim it exists in any release material.
- ACT/HOLD as a first-class decision rather than a failure mode: already demonstrated qualitatively in
  the matched-baseline numerical result (keyed_insertion: certificate matches estimator-default
  completion at 95% while moving some unsafe-ACT mass to HOLD, at a lower mean endpoint) -- this IS
  real evidence already in the repo, just not yet on real RGB-D data (the just-completed BOP run found
  0% certificate rate on real data at the current tolerances, so this qualitative story has NOT yet
  been reproduced on real sensor data).

**Action-selection (G_T/J_T-driven NEW_VIEW/REFINE/RESET) is scoped as a distinct, later project
phase** -- it is a materially new build (a policy over sensing actions, not just a stopping rule),
not a documentation or framing change. Do not fold it into the current release.

## Clarification, same day: why the BOP result is not yet on public GitHub

The founder separately checked github.com/morrocwi/task-conditioned-6d-pose-stop directly and (correctly)
did not find the BOP/RGB-D result there. This is by design, not an oversight: commit `fd65d27` (the real
BOP-LMO run) and the handoff commits around it were deliberately pushed only to the local Forgejo mirror,
held back from `origin` pending the independent-review workflow (`wf_934b7d58-e64`, launched this session)
that this repo's own culture requires before any public release. Once that review reports
`RELEASE READY: yes` (or its findings are resolved), the chair pushes `origin` and cuts the public release
with this result included in the evidence lineage -- not before. If the review is still running when a
reader checks GitHub, the honest answer is "not yet, on purpose, review in progress," not a bug.

## RELEASED — v0.7.0, 2026-09-09
Independent review (wf_934b7d58-e64) verdict PASS_WITH_WARNINGS: 2 confirmed blocks fixed (a /home
path leak scrubbed via git filter-branch on the 7 unpushed commits; AI-vendor attribution trailers
stripped from the real-data commit), 1 minor fixed (RESULT.md's coverage numerator corrected
32/40 -> 35/40, matching the stated 87.5%), 1 major disclosed honestly rather than silently fixed
(F4 -- the run-specific sampling/split parameters were committed alongside the results in the same
commit as lab_results.json, not frozen in a separate prior commit; a real process gap, now stated
plainly in the release notes rather than hidden). Chair verified the fixes independently, leak/
attribution-scanned the full push range clean, pushed origin+local, tagged v0.7.0, created the
GitHub release: https://github.com/morrocwi/task-conditioned-6d-pose-stop/releases/tag/v0.7.0.
Draft release-notes file deleted after publishing (content now lives in the GitHub release itself).

## Open TODO from this release
F4 (process gap, disclosed not fixed): future real-data runs should commit the frozen
config.json/split_manifest.json (or at least the sampling/split RULE, before any data-dependent
value is chosen) in a commit that predates seeing any final-test result, not alongside it. Track
this as a lab-protocol hardening item for the next real-backend run.

**F4 RESOLVED, 2026-09-09, run2**: commit `d9ac505f40df81344ffe0e3737cb1b51d36bf13e`
(2026-09-09T12:53:35+07:00) froze `lab/results/real-bop-lmo-2026-09-09-run2/config.json`,
`split_manifest.json`, `TOLERANCE_DECISION.md`, and `tolerance_derivation.json` (plus
`lab/derive_tolerances_from_train.py`) in a standalone commit, strictly before
`lab/run_real_system.py` was ever invoked against `test.jsonl` for run2 (that invocation and
`lab_results.json` came in a later, separate commit). No calibration/test statistic was read
before the freeze commit. This is the process fix F4 asked for.

## Second real-data run requested, 2026-09-09: "test dataset again"

Founder: "and. test dataset again" (terse, no further detail attached). Interpreted as: run another
real-data validation cycle on the same BOP LM-O dataset family, this time explicitly fixing the F4
process gap disclosed in v0.7.0's release notes (freeze config.json/split_manifest.json's sampling
and split RULE in its own commit, before any data-dependent value or final-test result is seen --
not alongside lab_results.json as last time), and address RESULT.md's own stated likely cause of
the 0% certificate rate: task tolerances were reused byte-identical from the numerical fixture
(lab/config.example.json) rather than derived for the real ape/driller objects' actual achievable
ICP accuracy in a 15-iteration budget. Re-running with the IDENTICAL tolerances would just reproduce
the same degenerate result with no new information -- not useful "testing again."

Plan for this run: (1) commit a frozen config (sampling rule + split rule + EITHER the same
tolerances, to test reproducibility of the same finding, OR properly re-derived tolerances for the
real objects, with the derivation method stated honestly and NOT tuned by peeking at final-test
outcomes) in its own commit, before opening final test; (2) re-run the full lab protocol; (3) report
whatever happens plainly, including if the result is unchanged, worse, or newly supports H2-H4.
Same honesty guardrails as the first run (CONTRIBUTING.md, evidence/ADVERSARIAL_REVIEW_RESPONSE.md,
this file's novelty/positioning constraints above) apply unchanged.

### Run2 executed, 2026-09-09

Decision: **re-derived** tolerances from TRAIN-split achievable ICP accuracy (not reused
identical), because reusing identical tolerances would deterministically reproduce run 1's own
already-stated diagnosis with no new information. Method: 75th percentile of the estimator-
convergence-stage absolute pose error across the 40 TRAIN episodes (unchanged data/split/seeds
from run 1), rounded up to 1mm/1deg, computed by `lab/derive_tolerances_from_train.py` reading
ONLY `train.jsonl`. Full rationale in
`lab/results/real-bop-lmo-2026-09-09-run2/TOLERANCE_DECISION.md`.

Frozen config/split/decision commit (predates the FINAL TEST evaluator run, fixing F4):
`d9ac505f40df81344ffe0e3737cb1b51d36bf13e`, 2026-09-09T12:53:35+07:00.

Result: unchanged from run 1 — `certificate_rate = 0.0`, `k_C<k_E` = 0/40, 100% HOLD, all three
tasks, held-out coverage 87.5% (identical, since coverage depends on calibration data/feature
model, not task tolerances). The loosened tolerances (2.6-2.8x translation, 1.4-1.8x rotation)
did raise the estimator's own completion rate (2.5%->32.5%, 2.5%->27.5%, 0%->10%), which rules
out run 1's own leading hypothesis that tolerances were simply too tight, and points instead at
the calibrated-envelope-construction step (observable feature vector / conformal calibration,
`cqts/safety.py`) as the more likely bottleneck. Non-inferiority FAILS more clearly than run 1
(LCB -49.1% / -43.9% / -23.7% vs run 1's -13.2% / -13.2% / -8.8%). Full detail:
`lab/results/real-bop-lmo-2026-09-09-run2/RESULT.md`. CLAIMS.md and README.md updated
accordingly; no existing OPEN/HOLD item retired.

Test suite, leak-scan, and push status: see the commit(s) following this handoff entry.

## Registered in glosa, 2026-09-09: GLS-2026-005

Founder: "ใช้ glosa สร้างสมมติฐานและยกระดับงาน". Registered lightweight (P02 intake + P08
diagnosis, full spine deliberately skipped, disclosed) at
`~/ANSE.ASIA/glosa/projects/GLS-2026-005_pose-stop-conformal-diagnosis/` (registry entry
`GLS-2026-005`, Blackbox Log `BBL-2026-09-09-249`). Diagnosis: the calibrated conformal quantile
`q~2.0008` (exp(q)~7.4x multiplicative inflation, rank 37/40) is the likely dominant driver of the
too-wide envelope on both real-data runs, not task tolerances (run2 already ruled that out).
Falsifiable next hypothesis, NOT yet tested: a per-stage or sequentially-valid conformal
aggregation (instead of the current whole-trajectory max over 15 stages x 6 coords) should shrink
q_alpha and could raise certificate rate above 0% on the identical data. Tiered `Dr`, explicitly
not `finite_diagnostic` -- P08 disciplines 2 (test the prediction) and 4 (independent
re-verification) are not yet applied; see the full diagnosis record for exactly what's missing.
This is the next real-data cycle to run when authorized (a third BOP-LMO run, same predeclare-
before-final-test discipline as run 2).

## Toledo registration + Coq witness, 2026-09-09

Per Toledo-first discipline, the base split-conformal machinery (C7-C9, already in
`cqts/safety.py`, retroactively) and the new Bonferroni-corrected multi-checkpoint fix designed
today are registered as Toledo proposals
(`~/ANSE.ASIA/toledo/registry/proposals/conformal_stopping_family.json`). The union-bound argument
behind the new fix is machine-checked, axiom-free
(`~/ANSE.ASIA/toledo/coq/canonical/PROP_CONF_03_union_bound.v`). Merge into Toledo's live registry
is paused on a founder ruling (does split-conformal prediction become a new Toledo root, like CMC)
-- this does not block implementing and testing the fix in THIS repo; cite the proposal ids
(PROP-CONF-01/02/03) in any code/doc reference until real weld/M codes are assigned.

Cleared-for-build spec for the next (third) real-data cycle: restrict certificate checks to K'<=4
predeclared checkpoint stages (feasibility bound computed for n=40 calibration episodes, alpha=0.1
-- see GLS-2026-005's diagnosis revision for the exact arithmetic), calibrate each checkpoint's
nonconformity score (coordinate-max only, no stage-aggregation) at level alpha/K' via the existing
`cqts/safety.py::safe_split_conformal_quantile`, and combine via the union bound proved in
PROP-CONF-03. This is a NEW code path (do not touch the existing C7-C9 joint construction used by
runs 1-2 -- keep both, so run 1/2 remain reproducible), predeclared and frozen before opening final
test, same discipline as run 2.

## Third real-data cycle launched, 2026-09-09
Building the Bonferroni-corrected multi-checkpoint fix (K'<=4, coordinate-max-only
nonconformity per checkpoint, cqts/safety.py's existing quantile function reused) as a NEW code
path alongside the existing whole-trajectory one (runs 1-2 stay reproducible). Will predeclare
config before final test, run on the identical BOP-LMO split, report per-checkpoint q vs the old
q~2.0008, update theory/FORMALIZATION_v1.md and GLS-2026-005's diagnosis with the outcome.

## Third real-data cycle executed, 2026-09-09: prediction refuted

Implemented `lab/multicheckpoint.py` + `cqts/safety.py::safe_multi_checkpoint_quantiles` /
`bonferroni_feasible_max_checkpoints` / `bonferroni_checkpoint_alpha` (PROP-CONF-03), additive
alongside the unmodified whole-trajectory path used by runs 1-2 (`lab/run_real_system.py --mode
whole_trajectory` vs `--mode bonferroni_multicheckpoint`). Full test suite (68 tests, including
new adversarial cases confirming K'=5 at n=40 fails closed to q=+infinity at every checkpoint)
passes.

Predeclared K'=4, checkpoints={4,8,12,15} (the largest feasible value per
`bonferroni_feasible_max_checkpoints(40,0.1)==4`), reusing run2's identical BOP-LMO data/split/ICP
backend and run2's TRAIN-derived tolerances unchanged, so the certificate-construction method is
the only varied factor. Frozen config/rationale committed (`681c710`,
2026-09-09T14:32:42+07:00) before `lab/run_real_system.py` was invoked against `test.jsonl` for
this run (invoked 2026-09-09T14:32:49+07:00, `lab_results.json` committed separately,
`a9a9a6a`). RATIONALE.md discloses one incidental smoke-test invocation against the real test
split during code development (no free parameter subsequently changed in response to it).

**Result: the falsifiable prediction is REFUTED.** The four per-checkpoint quantiles (q in
[2.085, 2.200]) came out 4.3-10.0% LARGER than the joint whole-trajectory q~2.0008 from runs 1-2,
not smaller. Certificate rate stayed at exactly 0.0 for all three tasks (100% HOLD), identical in
kind to runs 1-2. Likely cause: at n=40 calibration episodes, every checkpoint's
Bonferroni-corrected rank lands at 40/40 (the single largest calibration score), and this
per-checkpoint penalty outweighs the stage-aggregation penalty the construction removes. The
union-bound coverage guarantee itself held (all-checkpoints-covered rate 97.5% against a 90%
target) -- what is refuted is the specific quantitative prediction that this construction would
be cheaper than the joint one at this sample size, not the union-bound argument. Full detail:
`lab/results/real-bop-lmo-2026-09-09-run3/RESULT.md`. CLAIMS.md, README.md, and
theory/FORMALIZATION_v1.md (new C9b subsection) updated accordingly. GLS-2026-005's diagnosis in
glosa updated with the same outcome, its own commit in that repo.

## Fourth cycle authorized, 2026-09-09: spectrally-derived decay predictor
Founder: "หาขอบเขตล่างด้วยสมการ turbulance ที่เราพัฒนาไว้ได้" then "ทำเลย" -- use Toledo's own spectral
bounds (weld/M.40 Anderson-Morley, weld/M.42 Rayleigh -- both real Coq witnesses, proven today) plus
the Mohar/Fiedler diameter floor lambda_2>=4/(nD) (q_formal/M.07, currently a bare citation, not yet
Coq-proven) to replace C6's fitted log-linear error-scale predictor with a decay law derived from
the ICP normal-equations matrix's own condition number kappa=lambda_max/lambda_2:
rho_k<=(kappa-1)/(kappa+1), s_k,i := rho_k^(K-k) * current residual.
Toledo proposal: registry/proposals/spectral_decay_predictor.json (PROP-DECAY-01), parented to L_R,
weld/M.40.v1, weld/M.42.v1, q_formal/M.07.v1 -- honest caveats disclosed in the proposal itself
(the H_k~L_R identification is an ANALOGY not a proven fact; this is higher-risk than runs 2-3,
which only touched calibration, not the error estimate itself).

## Fourth cycle executed and pushed, 2026-09-09

Predeclaration commit `724c47a` (frozen config/split/RATIONALE + seed-identical
`bop_lmo_episodes_decay/{train,calibration,test}.jsonl`, before `test.jsonl` was ever evaluated),
result commit `bb90b68`. Both findings are refutations:

1. **Structural**: the PROP-DECAY-01 `H_k ~ graph-Laplacian L_R` identification does not hold for
   this real backend. `H_k=J^T J` is a fixed 6x6 SPD Gauss-Newton Hessian over the 6 pose degrees
   of freedom (exposed via a purely additive `lab/bop_icp_backend.py:normal_equations_H`
   diagnostic; the Kabsch update itself is unchanged) — it has no vertex/edge structure, so a
   graph diameter D is undefined for it, confirmed numerically against a separately-built real
   correspondence k-NN graph (n=400, D=16, `4/(nD)=0.000625`, unrelated to that same stage's
   `H_k` `lambda_min=0.1235`). The paired Coq-proof attempt on `q_formal/M.07` itself
   (`~/ANSE.ASIA/toledo`, separate task) also did not close in the time given — see that repo's
   own report for why (needs the full min-max/Courant-Fischer characterization of lambda_2, not
   available in this workspace's existing single-Rayleigh-pair spectral Coq idiom).
2. **Empirical**: using the correct quantity for `H_k` instead (ordinary matrix condition number,
   Kantorovich contraction bound, combined with the unmodified whole-trajectory C7-C9
   calibration), the resulting predictor's calibrated `q=3.298` is LARGER than both runs 1-2's
   `q≈2.0008` and every run3 Bonferroni per-checkpoint `q` (2.085-2.200). Certificate rate stays
   at exactly 0.0 for all three tasks (100% HOLD), the fourth cycle in a row to find this.

Full detail: `lab/results/real-bop-lmo-2026-09-09-run4/RESULT.md`. `CLAIMS.md`, `README.md`,
`theory/FORMALIZATION_v1.md` (new C6b section) updated. GLS-2026-005 (glosa, separate repo)
updated with the same outcome. Test suite: 81/81 pass (13 new tests in
`tests/test_decay_predictor.py`). Independent review (a materially separate agent pass, not
self-certification): verdict PASS — re-derived the core math, re-ran the evaluator and confirmed
a byte-identical `lab_results.json`, confirmed the predeclaration/F4 commit ordering, confirmed no
AI-vendor-attribution or local-path leaks in file content, confirmed no novelty/positioning
overclaims, confirmed the new `DecayModel` branch in `predicted_log_error` is genuinely additive
(re-ran run3's own invocation post-change and got a byte-identical result to what was already
committed). Pushed to both `local` (Forgejo) and `origin` (GitHub) — no release/tag cut this
cycle (no founder release request attached to this run).

## Fifth cycle authorized, 2026-09-09: H3 (native retained-sensitivity model)
Founder: "เราสร้างโมเดลมา แก้ปัญหาให้ certificate ทำงาน" (build the model, solve it so the
certificate works) -- read as proceeding with H3 (PROP-NATIVE-01/02, the native no-T* model),
since it is the only candidate that directly answers the founder's own redirect this session
(question what error IS under our philosophy; build from Toledo's own roots). H1/H2 remain
available candidates in GLOSA-PC-20260909-0005 if H3 also fails to certify.

Known, accepted, tracked limitation for THIS cycle (per theory/CONTINUUM_AUDIT_20260909.md item 1):
T_hat_k is still represented in SE(3), a continuum manifold -- H3 fixes the T* non-readout (no
ground truth used anywhere, even offline) but does NOT yet fix the deeper SE(3)-representation
continuum injection, which needs its own separate design pass. Proceeding anyway is a deliberate,
disclosed scope decision, not an oversight.

Implementation spec (from registry/proposals/native_retained_sensitivity.json, PROP-NATIVE-01/02):
1. Delta_k := retained update between consecutive pose estimates (already computable, no T*).
2. Reuse run 4's real H_k eigen-extraction (lab/decay_predictor.py or wherever it landed) to get
   real eigenpairs (lambda_j, v_j) of the ICP normal-equations Hessian at each stage -- this part
   of run 4 was NOT refuted (only the graph-diameter-based decay-rate use of it was); reuse the
   eigenvalues/eigenvectors directly.
3. Identify "data-justified-uncertain" directions: eigenvectors with small eigenvalue (near
   whatever floor is available -- q_formal/M.07 is still only a citation, so use an empirical/
   relative threshold, e.g. the smallest 1-2 eigenvalues, or a declared fraction of lambda_max,
   predeclared before final test, not tuned on results).
4. Perturbation magnitude along each such direction: proportional to 1/lambda_j, capped by a
   predeclared bound (do not let a near-zero eigenvalue produce an unbounded perturbation -- fail
   closed / HOLD if it does, matching this repo's own fail-closed culture).
5. NEW decision rule: ACT iff the task verdict O_T(That_k) is unchanged under EVERY predeclared
   perturbation along a data-justified-uncertain direction; otherwise CONTINUE/HOLD per budget.
   This REPLACES the calibrated-completion-set machinery (C7-C10) for this cycle's own decision
   path -- keep C7-C10 and the run1-4 code paths intact and reproducible, add this as a clearly
   separate new path (e.g. lab/native_sensitivity.py), do not delete or modify prior work.
6. No T* anywhere in the online decision. T* may be used ONLY afterward, as an external check on
   whether ACT-licensed episodes actually passed the task and CONTINUE/HOLD episodes were honestly
   uncertain -- exactly the same role T* already plays as an offline oracle in runs 1-4's own
   evaluation, never as a calibration input.
7. Predeclare the eigenvalue-threshold rule and perturbation-cap BEFORE opening final test (same
   TRAIN/CAL/TEST discipline, frozen config in its own commit, same as runs 2-4).
8. Run on the identical BOP-LMO data/split. Report honestly in
   lab/results/real-bop-lmo-2026-09-09-run5/ following RESULT_TEMPLATE.md -- whatever the outcome.
9. Update theory/FORMALIZATION_v1.md (this is the C1-C20/PROP-NATIVE material, a genuinely
   different branch from C21-C26 -- place it correctly, cross-reference
   docs/L_R_SPECTRAL_CEILING_AND_FLOOR.md and docs/NAVIER_STOKES_THROUGH_OUR_LENS.md's kappa-scaling
   warning since this construction also uses H_k eigenvalues). Update
   ~/ANSE.ASIA/glosa/projects/GLS-2026-005_pose-stop-conformal-diagnosis/DIAGNOSIS_HYPOTHESIS.md
   and the HYPOTHESIS_CANDIDATES_20260909.md file with the outcome against H3.
10. Novelty/positioning guardrails unchanged (never "first," never "we solve pose estimation").
    Full test suite, leak scan, independent review before pushing to origin (mirror runs 2-4).

## Fifth cycle executed, 2026-09-09: H3 fired ACT for the first time -- unsafely

Predeclaration commit `43954c09eedda7f3802adee6d83b559c836d19e6` (2026-09-09T16:03:50+07:00): froze
`lambda_ref` (median TRAIN lambda_min(H_k)) and `CAP` (p75 of TRAIN's local-dispersion-proxy norm)
via `lab/derive_native_threshold_from_train.py`, before `lab/run_native_sensitivity.py` was ever
invoked against `test.jsonl`. Result commit `dcd3343` (pushed to `local` only so far).

**This is the first of five real-data cycles to license ACT at all -- and it is unsafe.** ACT fired
on 100% of test episodes for all three declared tasks, always at the very first ICP stage (`k=0`,
before any refinement), and was WRONG on 92.5-100% of those episodes (114/120 task-episode pairs).
Diagnosed cause: the perturbation magnitude this construction produces is bounded well below the
real coarse-detector initial-pose error, so the invariance test answers whether a tiny wobble around
the current estimate is tolerable, not whether the estimate itself is close to correct. Full detail:
`lab/results/real-bop-lmo-2026-09-09-run5/RESULT.md`. `theory/FORMALIZATION_v1.md` (new C20b),
`CLAIMS.md`, `README.md` updated. `~/ANSE.ASIA/glosa` (GLS-2026-005) and `~/ANSE.ASIA/toledo`
(PROP-NATIVE-02 marked `refuted_unsafe`) updated in their own commits.

Per this cycle's own elevated bar (an unsafe-ACT finding, not a routine refutation), an independent
adversarial review (a materially separate agent pass, re-deriving the eigenvalue/perturbation math,
not just re-running the pipeline) was launched before any push to origin. Test suite: 95/95 pass
(81 pre-existing + 14 new in `tests/test_native_sensitivity.py`). Pushed to `local` (Forgejo) only;
push to `origin` (GitHub) is held pending that review's verdict, matching runs 2-4's own discipline
but with the higher bar this cycle's finding warrants.
