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
