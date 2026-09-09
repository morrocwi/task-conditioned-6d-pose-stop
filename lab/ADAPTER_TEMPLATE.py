"""Adapter template for an external iterative 6D pose backend.

Replace the marked functions with calls to your system. The public evaluator
consumes JSONL, so the backend itself does not need to depend on this project.
"""
from __future__ import annotations
import json
from pathlib import Path

def observable_features(backend_stage) -> list[float]:
    """Return ONLY online-observable quantities available before the stop decision.

    Examples: stage index, residual/RMS, pose update magnitude, backend score,
    ensemble dispersion, renderer/registration residual.

    Do not include ground-truth pose error or physical task outcome.
    """
    raise NotImplementedError

def estimator_stop(backend_stage) -> bool:
    """Return whether the backend's own stopping criterion fires at this stage."""
    raise NotImplementedError

def incremental_ms(backend_stage) -> float:
    """Measured wall-clock cost added by this stage, including required synchronization."""
    raise NotImplementedError

def abs_pose_error_6d(pose_estimate, pose_ground_truth) -> list[float]:
    """Independent oracle error [tx,ty,tz,rx,ry,rz] in metres/radians.

    Define and document the SE(3) error convention before collecting FINAL TEST.
    """
    raise NotImplementedError

def export_episode(episode_id, backend_stages, ground_truth_pose, output_file):
    stages=[]; errors=[]
    for k,stage in enumerate(backend_stages):
        stages.append({
            "k":k,
            "features":[float(x) for x in observable_features(stage)],
            "incremental_ms":float(incremental_ms(stage)),
            "estimator_stop":bool(estimator_stop(stage)),
        })
        errors.append([float(x) for x in abs_pose_error_6d(stage.pose, ground_truth_pose)])
    row={"episode_id":str(episode_id),"stages":stages,"oracle":{"abs_pose_error_6d":errors}}
    with Path(output_file).open("a",encoding="utf-8") as f:
        f.write(json.dumps(row)+"\n")
