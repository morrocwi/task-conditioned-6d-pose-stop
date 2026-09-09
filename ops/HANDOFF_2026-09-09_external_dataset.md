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
