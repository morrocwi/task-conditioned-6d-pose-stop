"""ADAPTER_TEMPLATE.py implementation for the real BOP LM-O ICP backend.

Wires lab/bop_icp_backend.py through the exact adapter functions required by
lab/ADAPTER_TEMPLATE.py, producing JSONL that validates against
lab/episode.schema.json. See lab/generate_bop_episodes.py for the driver that
loads the real dataset, runs the backend, and calls export_episode below.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from lab.bop_icp_backend import ICPStage, estimator_converged, orthonormalize
from lab.se3 import abs_pose_error_6d as se3_abs_pose_error_6d

FEATURE_EPS = 1e-9


def observable_features(stage: ICPStage) -> list[float]:
    """Online-observable ICP readouts only. No ground truth is referenced here.

    [k, log(rmse+eps), inlier_fraction, log(|delta_t|+eps), log(|delta_r|+eps),
     log(n_correspondences+1)] followed by the 6-component local dispersion
    proxy (log-transformed), same construction as experiments/numerical_icp.py's
    local_proxy but computed from the real depth correspondences.
    """
    base = [
        float(stage.k),
        math.log(stage.rmse + FEATURE_EPS),
        float(stage.inlier_fraction),
        math.log(stage.delta_t_norm + FEATURE_EPS),
        math.log(stage.delta_r_norm + FEATURE_EPS),
        math.log(stage.n_correspondences + 1),
    ]
    proxy = [math.log(float(x) + FEATURE_EPS) for x in stage.proxy]
    return base + proxy


def estimator_stop(stage: ICPStage) -> bool:
    """Real convergence criterion on the update magnitude (estimator-side stop)."""
    return bool(getattr(stage, "is_estimator_stop", False))


def incremental_ms(stage: ICPStage) -> float:
    """Measured wall-clock cost of this ICP iteration (query + Kabsch update)."""
    return float(stage.incremental_ms)


def abs_pose_error_6d(pose_estimate, pose_ground_truth) -> list[float]:
    """Independent oracle error using lab/se3.py's canonical convention.

    pose_estimate / pose_ground_truth are each a (R, t) tuple; converted to a
    4x4 homogeneous transform before calling the shared SE(3) helper.
    """
    R_e, t_e = pose_estimate
    R_g, t_g = pose_ground_truth
    T_e = np.eye(4)
    T_e[:3, :3] = orthonormalize(R_e)
    T_e[:3, 3] = t_e
    T_g = np.eye(4)
    T_g[:3, :3] = orthonormalize(R_g)
    T_g[:3, 3] = t_g
    return [float(x) for x in se3_abs_pose_error_6d(T_e, T_g)]


def mark_estimator_stop(stages: list[ICPStage]) -> None:
    """Mark exactly one stage's is_estimator_stop=True: the first stage k such that
    both stage k and the immediately preceding stage satisfy estimator_converged.
    Mirrors experiments/numerical_icp.py's estimator_stop_stage two-in-a-row rule.
    If no such pair exists, no stage is marked (run_real_system.py then falls back
    to the last stage as the estimator index, per its estimator_index()).
    """
    prev_small = False
    for s in stages[:-1]:
        small = estimator_converged(s)
        if small and prev_small:
            s.is_estimator_stop = True
            return
        prev_small = small


def export_episode(episode_id, backend_stages, ground_truth_pose, output_file):
    stages = []
    errors = []
    for k, stage in enumerate(backend_stages):
        stages.append({
            "k": k,
            "features": [float(x) for x in observable_features(stage)],
            "incremental_ms": float(incremental_ms(stage)),
            "estimator_stop": bool(estimator_stop(stage)),
        })
        errors.append([float(x) for x in abs_pose_error_6d((stage.R, stage.t), ground_truth_pose)])
    row = {"episode_id": str(episode_id), "stages": stages, "oracle": {"abs_pose_error_6d": errors}}
    with Path(output_file).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
