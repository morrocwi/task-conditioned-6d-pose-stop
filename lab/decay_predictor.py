"""Condition-number-bounded geometric decay predictor (PROP-DECAY-01).

Toledo proposal: ~/ANSE.ASIA/toledo/registry/proposals/spectral_decay_predictor.json.
Fourth real-data cycle (see ops/HANDOFF_2026-09-09_external_dataset.md, "Fourth
cycle authorized"). Replaces C6's fitted log-linear error-scale predictor
(theory/FORMALIZATION_v1.md) with a closed-form decay law derived from the
ICP normal-equations Hessian H_k = J^T J (lab/bop_icp_backend.py:normal_equations_H),
as a NEW code path -- C6's original log-linear model (lab/run_real_system.py:
fit_model/predicted_log_error) and the whole-trajectory (C7-C9) and
Bonferroni-checkpoint (C9b) calibration machinery are all left unmodified and
remain independently reproducible (runs 1-3).

Honest finding (read this before trusting kappa/rho numbers below)
--------------------------------------------------------------------
PROP-DECAY-01 proposed treating H_k as an instance of the same weighted-graph
-Laplacian family L_R := D_W - W that q_formal/M.07's diameter floor
lambda_2 >= 4/(n*D) is about, so that a graph node-count n and diameter D
could plug into that floor to bound H_k's own lambda_2 from below.

Having now exposed and inspected the real H_k this backend computes
(lab/bop_icp_backend.py:normal_equations_H), THIS ANALOGY DOES NOT HOLD
STRUCTURALLY for this backend:

  - H_k = J^T J is a FIXED 6x6 SPD matrix (one row/column per pose degree of
    freedom [omega_x,omega_y,omega_z,t_x,t_y,t_z]), built by summing n_k
    rank-3 contributions (one 3x6 Jacobian row-block per inlier correspondence).
    Its dimension never grows with n_k; more correspondences change H_k's
    six eigenvalues quantitatively but never its shape or its node/edge
    structure, because it HAS no node/edge structure -- it is a dense
    covariance-like Gram matrix over 6 abstract pose coordinates, not an
    adjacency/incidence structure over correspondence points.
  - q_formal/M.07's bound is about the Fiedler value (second-SMALLEST
    eigenvalue) of an n-node COMBINATORIAL graph Laplacian D_W - W, where n
    is the vertex count and D is a graph diameter (a path-length quantity
    that only makes sense on that same vertex/edge structure). H_k has no
    vertex set of size n_k and no path-length diameter at all -- "D" is
    simply undefined for a 6x6 dense Hessian.
  - The natural quantity for H_k is therefore the ORDINARY matrix condition
    number kappa = lambda_max(H_k) / lambda_min(H_k) (6 eigenvalues total,
    smallest of the six -- not a graph Fiedler value), which is exactly what
    the classical Gauss-Newton/steepest-descent contraction-rate bound
    rho <= (kappa-1)/(kappa+1) (Kantorovich inequality; cited, not
    Coq-derived here) already uses. That bound needs no graph structure and
    no q_formal/M.07 at all.

So: this module computes kappa_k/rho_k from H_k's ordinary 6-eigenvalue
condition number (ordinary matrix theory, ho graph object involved), and
SEPARATELY builds an actual n-node correspondence graph (a k-NN graph over
the inlier corresponded scene points) purely to report its own n and D
honestly, for comparison -- NOT because that graph's Laplacian has any
established relationship to H_k's spectrum. Task A (the Coq proof attempt on
q_formal/M.07 itself, ~/ANSE.ASIA/toledo) also did not close, independent of
this structural finding; even had it closed, the floor it would bound is a
different graph's Fiedler value, not lambda_min(H_k). Both an honest "not
applicable" comparison and a numeric n/D report are given in RESULT.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

FLOOR_EPS = 1e-12


@dataclass(frozen=True)
class KappaRho:
    lambda_min: float
    lambda_max: float
    kappa: float          # +inf if H_k is numerically singular (fail-closed)
    rho: float             # Kantorovich contraction-rate bound, clipped to [0, 1]
    degenerate: bool       # True if lambda_min ~ 0 (H_k could not certify contraction)


def kappa_rho_from_H(H: np.ndarray, singular_rtol: float = 1e-9) -> KappaRho:
    """Ordinary matrix condition number and Kantorovich contraction rate of a
    6x6 SPD (or PSD, if under-constrained) ICP normal-equations Hessian.

    Fail-closed: if H is (numerically) singular -- i.e. the six pose
    directions are not all locally observable from the current
    correspondences, a real and unremarkable situation early in ICP or with
    few inliers -- kappa=+inf and rho=1.0 (no contraction guaranteed), never
    a silently wrong finite number.
    """
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)  # symmetrize away float round-off before eigvalsh
    eigs = np.linalg.eigvalsh(H)
    lam_min = float(eigs[0])
    lam_max = float(eigs[-1])
    if not (math.isfinite(lam_min) and math.isfinite(lam_max)):
        return KappaRho(lam_min, lam_max, math.inf, 1.0, True)
    if lam_max <= 0.0 or lam_min <= singular_rtol * max(lam_max, FLOOR_EPS):
        return KappaRho(max(lam_min, 0.0), lam_max, math.inf, 1.0, True)
    kappa = lam_max / lam_min
    rho = (kappa - 1.0) / (kappa + 1.0)
    rho = min(max(rho, 0.0), 1.0)
    return KappaRho(lam_min, lam_max, float(kappa), float(rho), False)


def correspondence_graph_n_and_diameter(points: np.ndarray, k_nn: int = 6) -> dict:
    """Diagnostic only (see module docstring): n = number of inlier
    correspondence points, D = unweighted BFS diameter of a symmetrized k-NN
    graph over those points' scene-side positions at one ICP stage.

    This is a real, honestly-labelled graph (not H_k) -- reported to show,
    numerically, that n/D here grow with the correspondence count while
    H_k stays 6x6, i.e. that the two objects genuinely differ in kind.
    """
    n = int(len(points))
    if n < 2:
        return {"n": n, "diameter": None, "connected": False,
                "note": "fewer than 2 points; diameter undefined"}
    k = min(k_nn, n - 1)
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)  # includes self at column 0
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in idx[i, 1:]:
            j = int(j)
            adj[i].add(j)
            adj[j].add(i)
    # BFS from node 0; diameter = max over BFS eccentricities from every node
    # is O(n^2) -- fine for the small (<=800) per-stage correspondence counts
    # this backend uses, and only run on a handful of stages for reporting.
    def bfs_ecc(src):
        dist = {src: 0}
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        return dist

    reach0 = bfs_ecc(0)
    if len(reach0) < n:
        return {"n": n, "diameter": None, "connected": False,
                "note": f"k-NN graph (k={k}) disconnected: largest reach from node 0 = {len(reach0)}/{n}"}
    diam = 0
    for s in range(n):
        d = bfs_ecc(s)
        diam = max(diam, max(d.values()))
    return {"n": n, "diameter": int(diam), "connected": True, "k_nn": k}


def mohar_floor_4_over_nD(n: int, D) -> float | None:
    """q_formal/M.07's cited (NOT Coq-verified -- see Task A's report) floor
    4/(n*D), for honest side-by-side reporting only. Returns None if D is not
    a finite positive diameter (disconnected/degenerate graph -- the floor is
    only stated for connected graphs)."""
    if D is None or not (isinstance(D, (int, float)) and D > 0) or n <= 0:
        return None
    return 4.0 / (n * D)


@dataclass(frozen=True)
class DecayModel:
    """Duck-typed stand-in for lab/run_real_system.py's fitted `Model`
    (beta/mean/scale/error_floor): the decay predictor is closed-form, not
    fitted on TRAIN, so only `error_floor` (reused byte-identical from C6/C7-C9)
    is needed. lab/run_real_system.py:predicted_log_error() branches on
    isinstance(model, DecayModel) to call decay_predicted_log_error() below
    instead of the log-linear regression path -- C6's own function and the
    `Model`/`fit_model` path are untouched for any non-DecayModel caller.
    """
    error_floor: np.ndarray


def decay_predicted_log_error(stage: dict, axis: int, model: "DecayModel") -> float:
    """log s_{k,i} read from the "decay" field lab/adapter_bop.py's
    export_episode_decay() precomputed at generation time (k, K, rho,
    residual_axes are all known then; the floor is applied here, from
    `model.error_floor`, so the same floor config used by C6/C7-C9 governs
    this predictor too -- reused, not reimplemented, per the fourth-cycle plan.
    """
    d = stage.get("decay")
    if not isinstance(d, dict):
        raise ValueError(f"stage k={stage.get('k')} has no 'decay' field -- "
                          "was it generated by export_episode_decay?")
    rho = float(d["rho"])
    steps_remaining = max(0, int(d["K"]) - int(d["k"]))
    residual = float(d["residual_axes"][axis])
    floor = float(model.error_floor[axis])
    value = steps_remaining * math.log(max(rho, FLOOR_EPS)) + math.log(max(abs(residual), floor))
    if not math.isfinite(value):
        raise ValueError("decay_predicted_log_error produced a non-finite value")
    return value


def decay_log_error(kappa_rho: KappaRho, k: int, K: int, residual_axis: float, floor: float) -> float:
    """log s_{k,i} = (K-k) * log(rho_k + eps) + log(max(|residual|, floor)).

    residual_axis is the CURRENT stage's observable per-axis magnitude
    (this run uses lab/bop_icp_backend.py's local dispersion proxy -- the
    same online-observable quantity C6 already used as a raw feature -- as
    "current residual per coordinate"; no ground truth is used). floor is the
    per-axis error_floor also used by C6/C7-C9, reused unchanged.
    """
    rho = kappa_rho.rho
    steps_remaining = max(0, K - k)
    log_rho_term = steps_remaining * math.log(max(rho, FLOOR_EPS))
    log_resid_term = math.log(max(abs(float(residual_axis)), float(floor)))
    value = log_rho_term + log_resid_term
    if not math.isfinite(value):
        raise ValueError("decay_log_error produced a non-finite value")
    return value
