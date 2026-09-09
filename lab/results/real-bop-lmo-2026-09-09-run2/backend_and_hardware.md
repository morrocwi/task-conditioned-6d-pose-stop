# Backend and hardware — real-bop-lmo-2026-09-09-run2

Second real-data cycle on the same dataset/backend as
`lab/results/real-bop-lmo-2026-09-09/`. Everything below is byte-identical to
that run's `backend_and_hardware.md` unless marked "CHANGED".

## Backend

- Backend: in-repo numpy+scipy point-to-point ICP/Kabsch, `lab/bop_icp_backend.py` + `lab/adapter_bop.py` + `lab/generate_bop_episodes.py`. Unchanged code.
- Repository commit this run's frozen config/tolerance-decision was committed on top of: `d9ac505f40df81344ffe0e3737cb1b51d36bf13e` (parent `2b251f951fbb57c7423b6abc7b2bf0f994b43bcd`).
- Model/weights: none. Closed-form SVD/Kabsch rigid-transform update per iteration; correspondences via `scipy.spatial.cKDTree`. No learned/neural pose model of any kind.
- Correspondence/registration: point-to-point (not point-to-plane); no color/texture used, depth-only.
- Refinement budget: 15 iterations, fixed damping 0.65 per update. Unchanged.
- Coarse initializer: NOT a detector. A documented bounded random perturbation of real ground truth (`lab/bop_icp_backend.py:perturb_pose`) — translation += isotropic Gaussian sigma 15 mm/axis; rotation = ground truth composed with a random-axis rotation of angle Uniform[5, 20] degrees. Unchanged.
- Object segmentation for building the target point cloud: dataset-provided ground-truth `mask_visib`, not an online detector/segmentation network. This run tests the refinement stage only. Unchanged.

## Dataset

- BOP LineMOD-Occlusion (LM-O), `test_bop19` subset, scene `000002` (the dataset's only LM-O test scene). Unchanged.
- Source: `https://huggingface.co/datasets/bop-benchmark/lmo`, previously downloaded 2026-09-09; local cache reused (`lab/data/bop_lmo/`, not re-downloaded).
- Objects: `obj_id=1` (ape), `obj_id=8` (driller). Unchanged.
- **Split/frames/episodes: byte-identical to run 1** — same 120 episodes (60 frames x 2 objects), same 40/40/40 TRAIN/CALIBRATION/FINAL TEST split, same seeds (`lab/data/bop_lmo_episodes/{train,calibration,test,split_manifest}` files were not regenerated). This was a deliberate choice (see `TOLERANCE_DECISION.md`) to isolate tolerance choice as the only varied factor between run 1 and run 2.
- **CHANGED**: task tolerances (see `TOLERANCE_DECISION.md` and `tolerance_derivation.json`) — re-derived from TRAIN-split achievable ICP accuracy instead of reused from the numerical fixture.

## Hardware and software

- CPU: AMD Ryzen 7 4800H with Radeon Graphics (16 logical CPUs). Unchanged.
- GPU: an NVIDIA GeForce GTX 1650 Ti is present but **not used** — CPU-only numpy/scipy. Unchanged.
- Operating system: Linux 7.0.0-30-generic x86_64. Unchanged.
- Python 3.13.13 (conda-forge), numpy 2.4.6, scipy 1.17.1. Unchanged.
- Timing: `incremental_ms` is real measured per-iteration wall-clock, `trajectory_prefix_estimate` mode — no separate `online_policy_measured` dual-policy execution was run this cycle either, so no latency-reduction PASS/FAIL is emitted. Unchanged from run 1.
