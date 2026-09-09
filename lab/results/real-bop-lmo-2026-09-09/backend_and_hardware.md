# Backend and hardware — real-bop-lmo-2026-09-09

## Backend

- Backend: in-repo numpy+scipy point-to-point ICP/Kabsch, `lab/bop_icp_backend.py` + `lab/adapter_bop.py` + `lab/generate_bop_episodes.py`.
- Repository commit this was run against: `ed8735ad04c17efe1d1bc0a0cec4625071d2bdbb`.
- Model/weights: none. Closed-form SVD/Kabsch rigid-transform update per iteration; correspondences via `scipy.spatial.cKDTree`. No learned/neural pose model of any kind.
- Correspondence/registration: point-to-point (not point-to-plane); no color/texture used, depth-only.
- Refinement budget: 15 iterations, fixed damping 0.65 per update (unchanged from `experiments/numerical_icp.py`'s convention).
- Coarse initializer: NOT a detector. A documented bounded random perturbation of real ground truth (`lab/bop_icp_backend.py:perturb_pose`) — translation += isotropic Gaussian sigma 15 mm/axis; rotation = ground truth composed with a random-axis rotation of angle Uniform[5, 20] degrees.
- Object segmentation for building the target point cloud: dataset-provided ground-truth `mask_visib`, not an online detector/segmentation network. This run tests the refinement stage only.

## Dataset

- BOP LineMOD-Occlusion (LM-O), `test_bop19` subset, scene `000002` (the dataset's only LM-O test scene).
- Source: `https://huggingface.co/datasets/bop-benchmark/lmo` (`lmo_base.zip`, `lmo_models.zip`, `lmo_test_bop19.zip`), downloaded 2026-09-09.
- Objects: `obj_id=1` (ape), `obj_id=8` (driller).
- Camera intrinsics and per-frame ground-truth poses: dataset-provided `scene_camera.json` / `scene_gt.json`.
- 120 episodes total (60 frames x 2 objects), split 40/40/40 TRAIN/CALIBRATION/FINAL TEST by frame. Full selection/exclusion rule in `config.json` and `split_manifest.json`.

## Hardware and software

- CPU: AMD Ryzen 7 4800H with Radeon Graphics (16 logical CPUs).
- GPU: an NVIDIA GeForce GTX 1650 Ti is present in this machine but was **not used** — the ICP backend runs entirely on CPU via numpy/scipy; no CUDA/GPU code path exists in `lab/bop_icp_backend.py`.
- Operating system: Linux 7.0.0-30-generic x86_64 (glibc 2.43).
- Python: 3.13.13 (conda-forge build).
- numpy: 2.4.6.
- scipy: 1.17.1.
- Timing: `incremental_ms` in the exported episodes is real measured per-iteration wall-clock (`time.perf_counter()` around the cKDTree query + Kabsch SVD update), single-threaded process, no GPU synchronization involved. This is `trajectory_prefix_estimate` timing per `lab/README.md` — it is a real per-iteration cost, but no separate `online_policy_measured` dual-policy execution was run, so no latency-reduction PASS/FAIL is emitted (`config.json`'s `timing_mode`).
