"""Native retained-sensitivity ACT/CONTINUE/HOLD model (PROP-NATIVE-01/02).

Fifth real-data cycle (ops/HANDOFF_2026-09-09_external_dataset.md, "Fifth
cycle authorized, 2026-09-09: H3 (native retained-sensitivity model)"). This
is a GENUINELY DIFFERENT decision mechanism from runs 1-4: there is no fitted
error-shape model, no calibrated completion set, no conformal quantile, and
no comparison against a task tolerance via a predicted/derived error bound.
Runs 1-4's own code paths (`lab/run_real_system.py --mode
whole_trajectory|bonferroni_multicheckpoint|decay_predictor`) are untouched
and remain independently reproducible; this module is a clearly separate new
path, per the handoff's explicit instruction.

Mechanism
---------
At ICP stage k, the estimator's own current pose estimate T_hat_k = (R_k, t_k)
and the real ICP normal-equations Hessian H_k = J^T J (already computed by
`lab/bop_icp_backend.py:normal_equations_H`, reused unchanged) are both
already available online -- no ground truth anywhere in this section:

1. Eigendecompose H_k (`lab/decay_predictor.py:eigh_H`, reused unchanged --
   run4's own eigen-extraction code, not rewritten here). Select the SINGLE
   smallest eigenvalue lambda_min and its eigenvector v_min (predeclared
   rule: exactly one direction, the least-constrained one; not "all
   eigenvalues below a declared fraction of lambda_max" -- the simpler,
   least-discretionary of the two options the handoff offered, chosen to
   avoid any post-hoc argument about where a fraction threshold "should"
   sit).
2. Perturbation magnitude m = min(C / lambda_min, CAP), where CAP and C were
   frozen in `lab/derive_native_threshold_from_train.py` from TRAIN data
   only, before any calibration/test statistic was read (see that script's
   docstring and lab/results/real-bop-lmo-2026-09-09-run5/RATIONALE.md).
   Fail-closed: if lambda_min is non-finite or <= 0 (H_k numerically
   singular -- a real, unremarkable situation early in ICP or with few
   inliers, exactly as `lab/decay_predictor.py:kappa_rho_from_H` already
   treats it as fail-closed to kappa=+inf), m is simply CAP (the perturbation
   is bounded, never unbounded or silently treated as zero-uncertainty).
3. Two perturbed candidate poses are built: T_hat_k composed with
   +m*v_min and with -m*v_min (both signs of the eigenvector's arbitrary
   sign are tested, not one arbitrarily chosen side), using the SAME
   `rodrigues`/`compose` SE(3) helpers `lab/bop_icp_backend.py:run_icp`
   already uses for its own Kabsch update -- not a separately re-derived
   exponential map.
4. The task reader (`cqts.safety.fail_closed_task_pass`, reused unchanged)
   is invoked on the SE(3) deviation between T_hat_k and each candidate,
   computed via `lab/se3.py:abs_pose_error_6d` (reused unchanged; the same
   convention runs 1-4's own oracle scoring uses) -- i.e. "if the true pose
   were the perturbed candidate instead of T_hat_k, would the task reader's
   admissibility verdict change?" The unperturbed case is T_hat_k compared
   to itself, i.e. a zero deviation, which trivially reads as PASS for every
   task spec with strictly positive tolerances (see honest note below).
5. ACT iff the verdict is identical (PASS) for the unperturbed case and both
   perturbed candidates. Since the unperturbed case is a trivial PASS by
   construction (step 4), this reduces exactly to: ACT iff BOTH perturbed
   candidates also PASS. This reduction is stated here explicitly, not
   hidden -- it is a direct, disclosed consequence of using T_hat_k itself
   (zero self-deviation) as the "unperturbed pose" reading, the only
   ground-truth-free choice available for that half of the comparison.
   Otherwise: CONTINUE if stages remain in the budget, else HOLD.

No ground truth (T*) is used anywhere above. T* is used ONLY afterward, by
`lab/run_native_sensitivity.py`'s evaluator, in the exact same offline-oracle
role it already plays for runs 1-4: checking whether ACT-licensed episodes
were actually correct and CONTINUE/HOLD episodes were honestly uncertain.

Honest limitation carried over from run4 / theory/CONTINUUM_AUDIT_20260909.md
item 1: T_hat_k is still represented in SE(3), a continuum manifold. This
cycle fixes the T*-non-readout problem (H3's whole point) but does NOT fix
the deeper SE(3)-representation continuum injection -- a separate design pass,
not attempted here. The mixed units inside a single H_k eigenvector (three
components in radians, three in metres, perturbed by one shared scalar
magnitude m without separate nondimensionalization) is the same kind of
honesty gap already flagged for kappa in
docs/NAVIER_STOKES_THROUGH_OUR_LENS.md and disclosed again here rather than
silently assumed away.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cqts.safety import fail_closed_task_pass
from lab.bop_icp_backend import compose, orthonormalize, rodrigues
from lab.decay_predictor import eigh_H
from lab.se3 import abs_pose_error_6d as se3_abs_pose_error_6d


@dataclass(frozen=True)
class NativeConfig:
    lambda_ref: float
    cap: float

    @property
    def c_scale(self) -> float:
        return self.cap * self.lambda_ref


def select_least_constrained_direction(H: np.ndarray):
    """Predeclared rule: the SINGLE smallest eigenvalue/eigenvector of H_k.

    Returns (lambda_min, v_min) with v_min a unit 6-vector in [omega(3),
    t(3)] order (matching H_k's own column order, see
    lab/bop_icp_backend.py:normal_equations_H).
    """
    eigvals, eigvecs = eigh_H(H)
    return float(eigvals[0]), np.asarray(eigvecs[:, 0], dtype=float)


def perturbation_magnitude(lambda_min: float, cfg: NativeConfig) -> tuple[float, bool]:
    """m = min(C/lambda_min, CAP); fail-closed to CAP (never unbounded, never
    silently zero) if lambda_min is non-finite or <= 0. Returns (m, degenerate)."""
    if not math.isfinite(lambda_min) or lambda_min <= 0.0:
        return float(cfg.cap), True
    m = min(cfg.c_scale / lambda_min, cfg.cap)
    if not math.isfinite(m) or m < 0.0:
        return float(cfg.cap), True
    return float(m), False


def _to_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = orthonormalize(np.asarray(R, dtype=float))
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def perturbed_pose(R: np.ndarray, t: np.ndarray, v: np.ndarray, magnitude: float, sign: float):
    """T_hat_k composed with a signed perturbation along v (SE(3) local
    update), reusing the SAME `rodrigues`/`compose` helpers
    lab/bop_icp_backend.py:run_icp already uses for its own Kabsch update."""
    delta_w = sign * magnitude * np.asarray(v[:3], dtype=float)
    delta_t = sign * magnitude * np.asarray(v[3:], dtype=float)
    R_new, t_new = compose(rodrigues(delta_w), delta_t, np.asarray(R, dtype=float), np.asarray(t, dtype=float))
    return R_new, t_new


def deviation_axes(R_a, t_a, R_b, t_b) -> np.ndarray:
    """|SE(3) deviation| between two poses, in the SAME [tx,ty,tz,rx,ry,rz]
    convention runs 1-4's own oracle scoring uses (lab/se3.py), reused
    unchanged."""
    T_a = _to_homogeneous(R_a, t_a)
    T_b = _to_homogeneous(R_b, t_b)
    return np.asarray(se3_abs_pose_error_6d(T_a, T_b), dtype=float)


def stage_decision(R, t, H, spec: dict, cfg: NativeConfig) -> dict:
    """Evaluate the invariance test at one stage for one task spec.

    Returns a dict: act (bool), lambda_min, magnitude, degenerate,
    verdict_baseline, verdict_plus, verdict_minus, deviation_plus,
    deviation_minus -- everything needed for an honest, inspectable audit
    trail per stage, without ever touching ground truth.
    """
    lambda_min, v_min = select_least_constrained_direction(H)
    magnitude, degenerate = perturbation_magnitude(lambda_min, cfg)

    baseline_dev = np.zeros(6, dtype=float)
    verdict_baseline = bool(fail_closed_task_pass(spec, baseline_dev))

    R_plus, t_plus = perturbed_pose(R, t, v_min, magnitude, +1.0)
    R_minus, t_minus = perturbed_pose(R, t, v_min, magnitude, -1.0)
    dev_plus = deviation_axes(R, t, R_plus, t_plus)
    dev_minus = deviation_axes(R, t, R_minus, t_minus)
    verdict_plus = bool(fail_closed_task_pass(spec, dev_plus))
    verdict_minus = bool(fail_closed_task_pass(spec, dev_minus))

    act = verdict_baseline and verdict_plus and verdict_minus
    return {
        "act": bool(act),
        "lambda_min": lambda_min,
        "magnitude": magnitude,
        "degenerate": bool(degenerate),
        "verdict_baseline": verdict_baseline,
        "verdict_plus": verdict_plus,
        "verdict_minus": verdict_minus,
        "deviation_plus": [float(x) for x in dev_plus],
        "deviation_minus": [float(x) for x in dev_minus],
    }


def first_act_stage(stages: list[dict], spec: dict, cfg: NativeConfig):
    """Return (index, act, per_stage_audit) for the first stage where the
    invariance test licenses ACT, or (None, False, audit_list) if no stage in
    the budget does. Ground truth is absent from this function's signature."""
    audit = []
    for idx, stage in enumerate(stages):
        native = stage["native"]
        H = np.asarray(native["hessian"], dtype=float).reshape(6, 6)
        R = np.asarray(native["R"], dtype=float)
        t = np.asarray(native["t"], dtype=float)
        d = stage_decision(R, t, H, spec, cfg)
        audit.append(d)
        if d["act"]:
            return idx, True, audit
    return None, False, audit
