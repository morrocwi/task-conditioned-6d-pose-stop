#!/usr/bin/env python3
"""Generate real-backend episodes from BOP LM-O test scene 000002.

Dataset: BOP LineMOD-Occlusion (LM-O) test_bop19 subset, scene 000002 (the only
LM-O test scene), downloaded from huggingface.co/datasets/bop-benchmark/lmo
(lmo_base.zip, lmo_models.zip, lmo_test_bop19.zip). See
lab/results/real-bop-lmo-2026-09-09/backend_and_hardware.md for exact provenance.

Scope, predeclared and documented (not silently truncated):

  - Objects: obj_id 1 (ape) and obj_id 8 (driller) -- the two objects, among the
    eight annotated in LM-O scene 000002, with (a) very different size/shape
    (ape: 102mm diameter asymmetric organic shape; driller: 261mm diameter
    asymmetric tool shape) and (b) both present in the same large pool of frames,
    letting one frame set cover both objects without cherry-picking geometry.
  - Frames: only frames where BOTH objects are annotated as present (187 of 200
    scene frames qualify). From that pool, 60 frames are deterministically
    sampled (numpy Generator, seed 20260909) without replacement.
  - Episode = one (frame, object) pair. 60 frames x 2 objects = 120 episodes.
  - Object segmentation for building the target point cloud uses the dataset's
    own provided ground-truth `mask_visib` for that object in that frame. This
    is a real-sensor experiment for the REFINEMENT stage only: no online
    detector/segmentation network is executed, and this is stated as a limit,
    not disguised as an autonomous pipeline.
  - Initial pose for ICP is a documented bounded random perturbation of the
    real ground-truth pose (see lab/bop_icp_backend.py:perturb_pose),
    simulating a coarse detector stage that BOP raw data does not supply.

Split rule (frozen before any evaluator run): the 60 sampled frames are
themselves split 20/20/20 into TRAIN/CALIBRATION/FINAL TEST (numpy Generator,
seed 20260909, independent shuffle after the 60-frame draw) so that all
episodes from one frame (both objects) land in exactly one split -- avoiding
leakage from shared depth-noise realization across splits. Independent
sampling unit = frame (not object-episode), matching the disjointness
guarantee frozen in config.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.adapter_bop import export_episode, export_episode_decay, export_episode_native, mark_estimator_stop
from lab.bop_icp_backend import load_ply_vertices_m, perturb_pose, run_icp, subsample_points

DATA_DIR = ROOT / "lab" / "data" / "bop_lmo"
SCENE_DIR = DATA_DIR / "test" / "000002"
MODELS_DIR = DATA_DIR / "models"
OUT_DIR = ROOT / "lab" / "data" / "bop_lmo_episodes"
# run4 (PROP-DECAY-01): identical data/frames/seeds/ICP as above, written to a
# SEPARATE output directory via export_episode_decay (extra "decay" field per
# stage). Selected with --variant decay; default (no args, as run1-3 always
# invoked this script) is byte-identical to the original behavior.
OUT_DIR_DECAY = ROOT / "lab" / "data" / "bop_lmo_episodes_decay"
# run5 (PROP-NATIVE-01/02): identical data/frames/seeds/ICP as above, written
# to a SEPARATE output directory via export_episode_native (extra "native"
# field per stage carrying H_k + T_hat_k). Selected with --variant native.
OUT_DIR_NATIVE = ROOT / "lab" / "data" / "bop_lmo_episodes_native"

OBJECT_IDS = [1, 8]
N_FRAMES_TOTAL = 60
N_TRAIN_FRAMES = 20
N_CAL_FRAMES = 20
N_TEST_FRAMES = 20
FRAME_DRAW_SEED = 20260909
SPLIT_SHUFFLE_SEED = 20260909
MODEL_SUBSAMPLE_N = 400
MODEL_SUBSAMPLE_SEED_BASE = 700000
MAX_ITER = 15
DEPTH_SCALE_TO_M = 1e-3  # BOP LM-O ships depth in mm with depth_scale=1.0


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def backproject(depth_mm, mask, K, depth_scale):
    ys, xs = np.nonzero(mask)
    z = depth_mm[ys, xs].astype(float) * depth_scale * DEPTH_SCALE_TO_M
    valid = z > 0
    ys, xs, z = ys[valid], xs[valid], z[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def frame_mask_index(gt_frame, obj_id):
    for i, o in enumerate(gt_frame):
        if o["obj_id"] == obj_id:
            return i
    raise KeyError(f"object {obj_id} not present in frame")


def main(variant: str = "standard"):
    from PIL import Image

    export_fn = {"decay": export_episode_decay, "native": export_episode_native}.get(variant, export_episode)
    out_dir = {"decay": OUT_DIR_DECAY, "native": OUT_DIR_NATIVE}.get(variant, OUT_DIR)

    scene_gt = load_json(SCENE_DIR / "scene_gt.json")
    scene_cam = load_json(SCENE_DIR / "scene_camera.json")

    min_valid_depth_px = 300  # prefilter: frames where either object has too few
    # visible pixels WITH a nonzero real depth reading (real sensor dropout /
    # heavy occlusion) are excluded from the sampling pool entirely, before any
    # random draw -- not filtered post-hoc by ICP outcome.
    both_frames = []
    for f, objs in scene_gt.items():
        ids = {o["obj_id"] for o in objs}
        if not ids >= set(OBJECT_IDS):
            continue
        depth = np.array(Image.open(SCENE_DIR / "depth" / f"{int(f):06d}.png"))
        ok = True
        for oid in OBJECT_IDS:
            mi = frame_mask_index(objs, oid)
            mask = np.array(Image.open(SCENE_DIR / "mask_visib" / f"{int(f):06d}_{mi:06d}.png")) > 0
            valid = int(np.sum(mask & (depth > 0)))
            if valid < min_valid_depth_px:
                ok = False
                break
        if ok:
            both_frames.append(int(f))
    both_frames.sort()
    rng_draw = np.random.default_rng(FRAME_DRAW_SEED)
    drawn = rng_draw.choice(both_frames, size=N_FRAMES_TOTAL, replace=False)
    drawn = sorted(int(x) for x in drawn)

    rng_split = np.random.default_rng(SPLIT_SHUFFLE_SEED)
    shuffled = rng_split.permutation(drawn)
    train_frames = sorted(int(x) for x in shuffled[:N_TRAIN_FRAMES])
    cal_frames = sorted(int(x) for x in shuffled[N_TRAIN_FRAMES:N_TRAIN_FRAMES + N_CAL_FRAMES])
    test_frames = sorted(int(x) for x in shuffled[N_TRAIN_FRAMES + N_CAL_FRAMES:])
    assert len(set(train_frames) & set(cal_frames) & set(test_frames)) == 0
    assert len(train_frames) == N_TRAIN_FRAMES and len(cal_frames) == N_CAL_FRAMES and len(test_frames) == N_TEST_FRAMES

    models_m = {}
    for oid in OBJECT_IDS:
        verts = load_ply_vertices_m(MODELS_DIR / f"obj_{oid:06d}.ply")
        models_m[oid] = subsample_points(verts, MODEL_SUBSAMPLE_N, MODEL_SUBSAMPLE_SEED_BASE + oid)

    out_dir.mkdir(parents=True, exist_ok=True)
    split_map = {"train": train_frames, "calibration": cal_frames, "test": test_frames}
    manifest = {"objects": OBJECT_IDS, "frame_draw_seed": FRAME_DRAW_SEED,
                "split_shuffle_seed": SPLIT_SHUFFLE_SEED, "splits": {}}

    n_written = {"train": 0, "calibration": 0, "test": 0}
    for split_name, frames in split_map.items():
        out_path = out_dir / f"{split_name}.jsonl"
        if out_path.exists():
            out_path.unlink()
        episode_ids = []
        for frame_id in frames:
            frame_key = str(frame_id)
            K = np.asarray(scene_cam[frame_key]["cam_K"], float).reshape(3, 3)
            depth_scale = float(scene_cam[frame_key].get("depth_scale", 1.0))
            depth = np.array(Image.open(SCENE_DIR / "depth" / f"{frame_id:06d}.png"))
            gt_frame = scene_gt[frame_key]
            for oid in OBJECT_IDS:
                mi = frame_mask_index(gt_frame, oid)
                mask = np.array(Image.open(SCENE_DIR / "mask_visib" / f"{frame_id:06d}_{mi:06d}.png")) > 0
                scene_pts = backproject(depth, mask, K, depth_scale)
                episode_id = f"bopLMO-{split_name}-f{frame_id:06d}-o{oid}"
                if scene_pts.shape[0] < 50:
                    raise RuntimeError(f"{episode_id}: too few visible depth points ({scene_pts.shape[0]})")
                obj = gt_frame[mi]
                R_gt = np.asarray(obj["cam_R_m2c"], float).reshape(3, 3)
                t_gt = np.asarray(obj["cam_t_m2c"], float) * DEPTH_SCALE_TO_M
                seed_key = f"{split_name}-{frame_id}-{oid}".encode("utf-8")
                episode_seed = int.from_bytes(hashlib.sha256(seed_key).digest()[:4], "big")
                rng_ep = np.random.default_rng(episode_seed)
                R0, t0 = perturb_pose(rng_ep, R_gt, t_gt)
                scene_sub = subsample_points(scene_pts, 800, episode_seed)
                stages = run_icp(models_m[oid], scene_sub, R0, t0, max_iter=MAX_ITER)
                mark_estimator_stop(stages)
                export_fn(episode_id, stages, (R_gt, t_gt), out_path)
                episode_ids.append(episode_id)
                n_written[split_name] += 1
        manifest["splits"][split_name] = {"frames": frames, "episode_ids": episode_ids}

    Path(out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"episodes_written": n_written}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["standard", "decay", "native"], default="standard")
    args = parser.parse_args()
    main(args.variant)
