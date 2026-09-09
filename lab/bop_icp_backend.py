"""Real iterative point-to-point ICP backend against a real BOP RGB-D dataset.

Evidence boundary
------------------
EXECUTED: real depth-sensor point clouds (BOP LM-O test scene 000002), real object
meshes, real camera intrinsics, iterative point-to-point ICP/Kabsch registration on
numpy + scipy.spatial.cKDTree.

NOT EXECUTED: any learned/neural pose model, any coarse-detector/segmentation network
(object masks are taken from the dataset's own provided ground-truth visibility masks,
not detected online), any physical robot or contact physics.

The initial pose fed to ICP is a documented, bounded random perturbation of the real
ground-truth pose (see `perturb_pose`). This simulates a coarse detector stage that BOP
raw data does not provide; it is not a full autonomous detection pipeline.

Units inside this module: metres and radians (dataset ships mm; converted on load).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

MM_TO_M = 1e-3


# ---------------------------------------------------------------------------
# Minimal ASCII PLY reader (BOP LM/LM-O model files are ASCII, vertex x,y,z first).
# ---------------------------------------------------------------------------

def load_ply_vertices_m(path) -> np.ndarray:
    """Read vertex x,y,z (mm) from an ASCII PLY and return them in metres."""
    text = Path(path).read_text(encoding="ascii", errors="strict")
    lines = text.splitlines()
    if lines[0].strip() != "ply":
        raise ValueError(f"{path}: not a PLY file")
    fmt = None
    n_vertex = None
    header_end = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("format"):
            fmt = s.split()[1]
        elif s.startswith("element vertex"):
            n_vertex = int(s.split()[-1])
        elif s == "end_header":
            header_end = i
            break
    if fmt != "ascii":
        raise ValueError(f"{path}: only ascii PLY supported by this minimal parser, got {fmt}")
    if n_vertex is None or header_end is None:
        raise ValueError(f"{path}: malformed PLY header")
    verts = np.empty((n_vertex, 3), dtype=float)
    for i in range(n_vertex):
        parts = lines[header_end + 1 + i].split()
        verts[i] = (float(parts[0]), float(parts[1]), float(parts[2]))
    return verts * MM_TO_M


def subsample_points(points: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    if len(points) <= n_target:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=n_target, replace=False)
    return points[idx]


# ---------------------------------------------------------------------------
# SE(3) helpers (local copies; consistent with lab/se3.py rotation-vector convention)
# ---------------------------------------------------------------------------

def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], float)


def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th
    K = skew(k)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def rotvec(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-10:
        return np.zeros(3)
    s = math.sin(th)
    if abs(s) < 1e-10:
        vals, vecs = np.linalg.eig(R)
        axis = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        axis /= np.linalg.norm(axis)
        return axis * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * s)
    return v * th


def compose(R1, t1, R2, t2):
    return R1 @ R2, R1 @ t2 + t1


def apply(R, t, P):
    return (R @ P.T).T + t


def best_fit_kabsch(A, B):
    """Rigid transform mapping A onto B (SVD/Kabsch)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cb - R @ ca


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix to the nearest proper rotation (SVD).

    BOP ground-truth `cam_R_m2c` matrices are stored to finite float precision
    and are not exactly orthonormal (observed orthogonality residual up to
    ~3e-4 on this dataset, exceeding lab/se3.py's 1e-5 validation tolerance).
    This is real-dataset floating-point imprecision, not part of the pose
    estimate under test; it is corrected the same way for ground truth and for
    every ICP stage output before oracle error computation.
    """
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def perturb_pose(rng, R_gt, t_gt, trans_sigma_m=0.015, rot_low_deg=5.0, rot_high_deg=20.0):
    """Bounded random perturbation of ground truth simulating a coarse detector.

    Translation: isotropic Gaussian noise, sigma=15 mm per axis.
    Rotation: a random axis with angle drawn Uniform[5, 20] degrees, composed
    with the ground-truth rotation (NOT a per-axis additive Euler perturbation).
    Explicitly documented; not a full detection pipeline.
    """
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = math.radians(rng.uniform(rot_low_deg, rot_high_deg))
    R_delta = rodrigues(axis * angle)
    R0 = R_delta @ R_gt
    t0 = t_gt + rng.normal(0.0, trans_sigma_m, size=3)
    return R0, t0


# ---------------------------------------------------------------------------
# Local linearized dispersion proxy (observable; no ground truth). Same construction
# as experiments/numerical_icp.py's local_proxy, applied to the real correspondences.
# ---------------------------------------------------------------------------

def local_proxy(model, R, t, corr):
    X = apply(R, t, model)
    r = (X - corr).reshape(-1)
    J = np.vstack([np.hstack((-skew(x), np.eye(3))) for x in X])
    dof = max(1, len(r) - 6)
    sigma2 = float((r @ r) / dof)
    cov = sigma2 * np.linalg.pinv(J.T @ J, rcond=1e-10)
    sd = np.sqrt(np.maximum(0.0, np.diag(cov)))
    return np.r_[sd[3:], sd[:3]]


def normal_equations_H(model, R, t, corr):
    """ICP normal-equations (Gauss-Newton) Hessian H = J^T J at the current
    linearization point, over the same inlier correspondences local_proxy()
    above uses (identical J: residual r = R*p+t - q, J_i = [-skew(R*p_i), I_3],
    columns ordered [omega(3), t(3)]).

    This is a genuinely computed diagnostic quantity: it is what local_proxy()
    already inverts internally (J.T @ J) to build its covariance proxy, just
    exposed here directly rather than only its diagonal-inverse. It is NOT
    part of the Kabsch update itself (best_fit_kabsch is a closed-form SVD
    solve, not an iterative linearized normal-equations solve) -- H_k is a
    post-hoc diagnostic of the same correspondences, added per
    registry/proposals/spectral_decay_predictor.json (PROP-DECAY-01,
    ~/ANSE.ASIA/toledo) to test the H_k ~ graph-Laplacian analogy against
    this real backend. See lab/decay_predictor.py for the honest finding on
    whether that analogy holds structurally here.
    """
    X = apply(R, t, model)
    J = np.vstack([np.hstack((-skew(x), np.eye(3))) for x in X])
    return J.T @ J


@dataclass
class ICPStage:
    k: int
    R: np.ndarray
    t: np.ndarray
    delta_t_norm: float
    delta_r_norm: float
    rmse: float
    inlier_fraction: float
    n_correspondences: int
    proxy: np.ndarray
    incremental_ms: float
    is_estimator_stop: bool = False
    # Optional diagnostic ICP normal-equations Hessian H_k = J^T J (6x6 SPD),
    # added for PROP-DECAY-01 (see normal_equations_H() above). Defaulted to
    # None so any existing caller/test that builds an ICPStage positionally
    # without this field is unaffected.
    hessian: np.ndarray | None = None


def run_icp(model_pts, scene_pts, R0, t0, max_iter=20, damping=0.65, inlier_mult=3.0):
    """Real point-to-point ICP: cKDTree correspondences + Kabsch SVD update per step.

    inlier_mult: a correspondence is an "inlier" if its distance is <= inlier_mult
    times the median correspondence distance at that iteration (robust, observable,
    does not use ground truth).
    """
    tree = cKDTree(scene_pts)
    R, t = R0.copy(), t0.copy()
    stages = []
    for k in range(max_iter + 1):
        tick = time.perf_counter()
        X = apply(R, t, model_pts)
        dist, idx = tree.query(X, k=1)
        corr = scene_pts[idx]
        rmse = float(np.sqrt(np.mean(dist ** 2)))
        med = float(np.median(dist)) if len(dist) else float("inf")
        inliers = dist <= inlier_mult * max(med, 1e-9)
        inlier_fraction = float(np.mean(inliers))
        proxy_model = model_pts[inliers] if inliers.sum() >= 10 else model_pts
        proxy_corr = corr[inliers] if inliers.sum() >= 10 else corr
        proxy = local_proxy(proxy_model, R, t, proxy_corr)
        H = normal_equations_H(proxy_model, R, t, proxy_corr)
        if k == max_iter or inliers.sum() < 10:
            elapsed_ms = (time.perf_counter() - tick) * 1000.0
            stages.append(ICPStage(k, R.copy(), t.copy(), 0.0, 0.0, rmse, inlier_fraction,
                                    int(inliers.sum()), proxy, elapsed_ms, hessian=H))
            break
        dR, dtv = best_fit_kabsch(X[inliers], corr[inliers])
        w = rotvec(dR)
        delta_t = damping * dtv
        delta_w = damping * w
        R_new, t_new = compose(rodrigues(delta_w), delta_t, R, t)
        elapsed_ms = (time.perf_counter() - tick) * 1000.0
        stages.append(ICPStage(k, R.copy(), t.copy(), float(np.linalg.norm(delta_t)),
                                float(np.linalg.norm(delta_w)), rmse, inlier_fraction,
                                int(inliers.sum()), proxy, elapsed_ms, hessian=H))
        R, t = R_new, t_new
    return stages


def estimator_converged(stage: ICPStage, trans_tol_m=0.0005, rot_tol_rad=math.radians(0.1)) -> bool:
    """Real convergence criterion on the update magnitude (estimator-side stop)."""
    return stage.delta_t_norm <= trans_tol_m and stage.delta_r_norm <= rot_tol_rad
