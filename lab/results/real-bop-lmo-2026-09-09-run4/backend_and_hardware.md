# Backend and hardware — real-bop-lmo-2026-09-09-run3

Third real-data cycle on the same dataset/backend as `lab/results/real-bop-lmo-2026-09-09/`
(run 1) and `lab/results/real-bop-lmo-2026-09-09-run2/` (run 2). Everything below is
byte-identical to run 2's `backend_and_hardware.md` unless marked "CHANGED". This run changes
only the CERTIFICATE-CONSTRUCTION method (Bonferroni-corrected multi-checkpoint conformal band,
PROP-CONF-03, `lab/multicheckpoint.py`), not the data, split, backend, or task tolerances.

## Backend

- Backend: in-repo numpy+scipy point-to-point ICP/Kabsch, `lab/bop_icp_backend.py` +
  `lab/adapter_bop.py` + `lab/generate_bop_episodes.py`. Unchanged code, unchanged episodes.
- Repository commit this run's frozen config/checkpoint-decision was committed on top of:
  `cfe50a40d19223ba6d4652fa2e8ff6f236d2fdd4` (adds `lab/multicheckpoint.py` and
  `cqts.safety`'s Bonferroni functions; code+tests only, no results, no config).
- Model/weights: none. Closed-form SVD/Kabsch rigid-transform update per iteration;
  correspondences via `scipy.spatial.cKDTree`. No learned/neural pose model of any kind. Unchanged.
- Correspondence/registration: point-to-point (not point-to-plane); depth-only. Unchanged.
- Refinement budget: 15 iterations, fixed damping 0.65 per update. Unchanged.
- Coarse initializer: NOT a detector. A documented bounded random perturbation of real ground
  truth (`lab/bop_icp_backend.py:perturb_pose`). Unchanged.
- Object segmentation: dataset-provided ground-truth `mask_visib`, not an online
  detector/segmentation network. This run tests the refinement stage only. Unchanged.

## CHANGED this run — certificate construction, not data

- **CHANGED**: the certificate is now built by `lab/multicheckpoint.py`
  (`lab/run_real_system.py --mode bonferroni_multicheckpoint`), a NEW code path alongside the
  existing whole-trajectory construction (unchanged, still used by `--mode whole_trajectory`,
  which reproduces runs 1-2 unchanged).
- Nonconformity score is now computed independently at each of K'=4 predeclared checkpoint
  stages, as a max over the 6 pose COORDINATES ONLY (never over stages) — see `RATIONALE.md`.
- Each checkpoint is calibrated at level `alpha/K'=0.025` via the SAME
  `cqts.safety.safe_split_conformal_quantile` function used by the whole-trajectory path (called
  once per checkpoint with the smaller alpha), reused via
  `cqts.safety.safe_multi_checkpoint_quantiles`.
- Task readers, task tolerances, `alpha=0.1`, `error_floor`, `inference` block, `timing_mode`:
  all UNCHANGED from run 2 (see `RATIONALE.md` for why).

## Dataset

- BOP LineMOD-Occlusion (LM-O), `test_bop19` subset, scene `000002`. Unchanged.
- Source: `https://huggingface.co/datasets/bop-benchmark/lmo`; local cache reused
  (`lab/data/bop_lmo/`, not re-downloaded).
- Objects: `obj_id=1` (ape), `obj_id=8` (driller). Unchanged.
- **Split/frames/episodes: byte-identical to runs 1 and 2** — same 120 episodes (60 frames x 2
  objects), same 40/40/40 TRAIN/CALIBRATION/FINAL TEST split, same seeds
  (`lab/data/bop_lmo_episodes/{train,calibration,test,split_manifest}` files were not
  regenerated; `split_manifest.json` in this directory is a copy of run 2's, unchanged). This
  isolates the certificate-construction method (PROP-CONF-03 vs. the joint whole-trajectory
  construction) as the ONLY varied factor across all three real-data cycles.

## Hardware and software

- CPU: AMD Ryzen 7 4800H with Radeon Graphics (16 logical CPUs). Unchanged.
- GPU: present but **not used** — CPU-only numpy/scipy. Unchanged.
- Operating system: Linux 7.0.0-30-generic x86_64. Unchanged.
- Python 3.13.13 (conda-forge), numpy 2.4.6, scipy 1.17.1. Unchanged.
- Timing: `incremental_ms` is real measured per-iteration wall-clock, `trajectory_prefix_estimate`
  mode — no separate `online_policy_measured` dual-policy execution was run this cycle either.
  Unchanged from runs 1-2.
