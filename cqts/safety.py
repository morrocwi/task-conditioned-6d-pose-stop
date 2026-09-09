from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import numpy as np


class CertificateNumericsError(ValueError):
    """Raised when certificate inputs are malformed or numerically invalid."""


def safe_split_conformal_quantile(scores: Iterable[float], alpha: float):
    """Finite-sample split-conformal order statistic with +inf augmentation.

    For n calibration scores, rank = ceil((n+1)(1-alpha)). If rank == n+1,
    the augmented order statistic is +infinity. This yields an uninformative
    envelope and therefore a fail-closed HOLD rather than silently overstating
    finite-sample coverage.
    """
    x = np.asarray(list(scores), dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise CertificateNumericsError("scores must be a non-empty 1-D sequence")
    if not (0.0 < float(alpha) < 1.0) or not math.isfinite(float(alpha)):
        raise CertificateNumericsError("alpha must be finite and in (0,1)")
    if not np.all(np.isfinite(x)):
        raise CertificateNumericsError("calibration scores must all be finite")
    x = np.sort(x)
    n = int(x.size)
    rank = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    if not (1 <= rank <= n + 1):
        raise CertificateNumericsError("invalid conformal rank")
    if rank == n + 1:
        return math.inf, rank
    return float(x[rank - 1]), rank


def bonferroni_feasible_max_checkpoints(n: int, alpha: float) -> int:
    """Largest K' for which every checkpoint stays feasible at level alpha/K'.

    A predeclared checkpoint calibrated via `safe_split_conformal_quantile` at
    level alpha/K' is feasible (does not fail-close to +infinity) exactly when
    `ceil((n+1)(1-alpha/K')) <= n`. As K' grows, alpha/K' shrinks, so the
    required rank `ceil((n+1)(1-alpha/K'))` is non-decreasing in K' -- once a
    K' becomes infeasible, every larger K' is infeasible too. This walks K'
    upward and returns the last feasible value (0 if even K'=1 is infeasible
    for this n and alpha, which would only happen for very small n).

    This is PROP-CONF-03's feasibility constraint
    (~/ANSE.ASIA/toledo/registry/proposals/conformal_stopping_family.json),
    verified for n=40, alpha=0.1 to give exactly K'<=4 feasible, K'=5
    infeasible -- see GLS-2026-005's diagnosis revision in glosa.
    """
    n = int(n)
    if n < 1:
        raise CertificateNumericsError("n must be >= 1")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0) or not math.isfinite(alpha):
        raise CertificateNumericsError("alpha must be finite and in (0,1)")
    best = 0
    k = 1
    while k <= n + 1:
        per_checkpoint_alpha = alpha / k
        rank = int(math.ceil((n + 1) * (1.0 - per_checkpoint_alpha)))
        if rank <= n:
            best = k
            k += 1
        else:
            break
    return best


def bonferroni_checkpoint_alpha(alpha: float, num_checkpoints: int) -> float:
    """Per-checkpoint alpha level for a Bonferroni/union-bound multi-checkpoint
    conformal band (PROP-CONF-03): splits the overall level alpha evenly across
    `num_checkpoints` predeclared checkpoints so the union bound (Boole's
    inequality) keeps the combined miscoverage probability across all
    checkpoints <= alpha. Machine-checked, axiom-free:
    ~/ANSE.ASIA/toledo/coq/canonical/PROP_CONF_03_union_bound.v
    (`finite_union_bound`, `bonferroni_checkpoints`).
    """
    if not (0.0 < float(alpha) < 1.0) or not math.isfinite(float(alpha)):
        raise CertificateNumericsError("alpha must be finite and in (0,1)")
    if int(num_checkpoints) < 1:
        raise CertificateNumericsError("num_checkpoints must be >= 1")
    return float(alpha) / int(num_checkpoints)


def safe_multi_checkpoint_quantiles(
    scores_by_checkpoint: Mapping[int, Iterable[float]],
    alpha: float,
):
    """Bonferroni-corrected multi-checkpoint conformal quantiles (PROP-CONF-03).

    Given calibration nonconformity scores computed INDEPENDENTLY at each of
    K' predeclared checkpoint stages (each score a max over the 6 pose
    coordinates ONLY -- never over stages, unlike the whole-trajectory
    PROP-CONF-02 construction used by `safe_split_conformal_quantile` calls in
    runs 1-2), this calibrates every checkpoint at level alpha/K' by calling
    the SAME `safe_split_conformal_quantile` function used by the existing
    whole-trajectory path -- the conformal order-statistic logic itself is
    never reimplemented, only invoked once per checkpoint with a smaller
    alpha. Do not use this to replace `safe_split_conformal_quantile` calls
    made for the existing whole-trajectory construction; both code paths must
    remain available so runs 1-2 stay reproducible.

    By the union bound (Boole's inequality; machine-checked axiom-free,
    `~/ANSE.ASIA/toledo/coq/canonical/PROP_CONF_03_union_bound.v`, theorems
    `finite_union_bound` / `bonferroni_checkpoints`), the resulting per-
    checkpoint envelopes combine to an overall >= 1-alpha whole-trajectory
    coverage guarantee without the joint stage-aggregation of PROP-CONF-02.

    Returns {checkpoint_k: (q, rank)}, one entry per checkpoint, in the same
    (q, rank) shape `safe_split_conformal_quantile` itself returns. Fails
    closed exactly the same way the underlying function does: whenever a
    checkpoint's Bonferroni-corrected rank exceeds n (K' predeclared too
    large for n calibration episodes -- see
    `bonferroni_feasible_max_checkpoints`), that checkpoint's q is
    +infinity, never silently something else.
    """
    checkpoints = list(scores_by_checkpoint.keys())
    if not checkpoints:
        raise CertificateNumericsError("scores_by_checkpoint must be non-empty")
    per_checkpoint_alpha = bonferroni_checkpoint_alpha(alpha, len(checkpoints))
    out = {}
    for k in checkpoints:
        out[k] = safe_split_conformal_quantile(scores_by_checkpoint[k], per_checkpoint_alpha)
    return out


def safe_exp_error_bound(predicted_log_error: float, q: float, floor: float) -> float:
    """Convert log error prediction + conformal scale into a fail-closed bound.

    Deliberate +inf q returns +inf. Corrupt numerics never collapse to zero.
    Finite negative bounds produced by subtracting the positive floor are
    clamped to zero only after all upstream quantities have been validated.
    """
    p = float(predicted_log_error)
    q = float(q)
    floor = float(floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise CertificateNumericsError("error floor must be finite and > 0")
    if q == math.inf:
        return math.inf
    if not math.isfinite(q):
        raise CertificateNumericsError("q must be finite or +inf")
    if not math.isfinite(p):
        raise CertificateNumericsError("predicted log error must be finite")
    z = p + q
    if not math.isfinite(z):
        return math.inf
    try:
        raw = math.exp(z) - floor
    except OverflowError:
        return math.inf
    if not math.isfinite(raw):
        return math.inf
    return max(0.0, raw)


def validate_task_spec(spec: dict, dim: int = 6) -> None:
    kind = spec.get("kind")
    if kind not in {"box", "l1"}:
        raise CertificateNumericsError(f"unsupported task kind {kind!r}")
    mask = spec.get("mask")
    tol = spec.get("tol")
    if not isinstance(mask, (list, tuple)) or not mask:
        raise CertificateNumericsError("task mask must be a non-empty list/tuple")
    if not isinstance(tol, (list, tuple, np.ndarray)) or len(tol) != dim:
        raise CertificateNumericsError(f"task tol must contain {dim} values")
    seen = set()
    for i in mask:
        if not isinstance(i, (int, np.integer)) or not (0 <= int(i) < dim):
            raise CertificateNumericsError("task mask index out of range")
        if int(i) in seen:
            raise CertificateNumericsError("task mask contains duplicate index")
        seen.add(int(i))
        t = float(tol[int(i)])
        if not math.isfinite(t) or t <= 0.0:
            raise CertificateNumericsError("masked task tolerances must be finite and > 0")


def fail_closed_task_pass(spec: dict, bounds: Sequence[float]) -> bool:
    """Robust PASS for monotone absolute-error readers; invalid inputs HOLD."""
    try:
        validate_task_spec(spec)
        b = np.asarray(bounds, dtype=float)
        if b.shape != (6,):
            return False
        if np.any(np.isnan(b)) or np.any(b < 0.0):
            return False
        mask = [int(i) for i in spec["mask"]]
        tol = np.asarray(spec["tol"], dtype=float)
        if spec["kind"] == "box":
            return bool(all(float(b[i]) <= float(tol[i]) for i in mask))
        total = 0.0
        for i in mask:
            bi = float(b[i])
            if not math.isfinite(bi):
                return False
            total += bi / float(tol[i])
        return bool(total <= 1.0)
    except (ValueError, TypeError, CertificateNumericsError, FloatingPointError):
        return False


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return -math.inf
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def _sum_binom_range(lo: int, hi: int, n: int, p: float) -> float:
    if lo > hi:
        return 0.0
    logs = [_log_binom_pmf(k, n, p) for k in range(lo, hi + 1)]
    m = max(logs)
    if m == -math.inf:
        return 0.0
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return _sum_binom_range(0, k, n, p)


def _binom_sf_ge(k: int, n: int, p: float) -> float:
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return _sum_binom_range(k, n, n, p)


def clopper_pearson_lower(k: int, n: int, tail_alpha: float) -> float:
    if not (0 <= k <= n) or n <= 0:
        raise ValueError("require 0 <= k <= n and n > 0")
    if not (0.0 < tail_alpha < 1.0):
        raise ValueError("tail_alpha must be in (0,1)")
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(70):
        mid = (lo + hi) / 2.0
        if _binom_sf_ge(k, n, mid) < tail_alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson_upper(k: int, n: int, tail_alpha: float) -> float:
    if not (0 <= k <= n) or n <= 0:
        raise ValueError("require 0 <= k <= n and n > 0")
    if not (0.0 < tail_alpha < 1.0):
        raise ValueError("tail_alpha must be in (0,1)")
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(70):
        mid = (lo + hi) / 2.0
        if _binom_cdf(k, n, mid) > tail_alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def paired_binary_noninferiority(
    candidate: Sequence[int | bool],
    comparator: Sequence[int | bool],
    margin: float,
    confidence_level: float = 0.95,
    minimum_n: int = 1,
):
    """Conservative finite-sample paired binary non-inferiority decision.

    Let p10=P(candidate=1, comparator=0), p01=P(candidate=0, comparator=1),
    so the paired completion difference is delta=p10-p01. We construct exact
    one-sided Clopper-Pearson marginal bounds for p10 and p01 with Bonferroni
    allocation, giving a conservative lower confidence bound for delta.
    """
    a = np.asarray(candidate, dtype=int)
    b = np.asarray(comparator, dtype=int)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) == 0:
        raise ValueError("paired binary arrays must be non-empty, 1-D, equal length")
    if np.any((a != 0) & (a != 1)) or np.any((b != 0) & (b != 1)):
        raise ValueError("paired outcomes must be binary")
    n = int(len(a))
    if not (0.0 < float(margin) < 1.0):
        raise ValueError("margin must be in (0,1)")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError("confidence_level must be in (0,1)")
    if int(minimum_n) < 1:
        raise ValueError("minimum_n must be >= 1")

    gains = int(np.sum((a == 1) & (b == 0)))
    losses = int(np.sum((a == 0) & (b == 1)))
    alpha = 1.0 - float(confidence_level)
    tail = alpha / 2.0
    gain_lower = clopper_pearson_lower(gains, n, tail)
    loss_upper = clopper_pearson_upper(losses, n, tail)
    lower = gain_lower - loss_upper
    diff = float(np.mean(a - b))
    enough_n = n >= int(minimum_n)
    return {
        "method": "paired_discordance_bonferroni_clopper_pearson_lower_bound",
        "confidence_level": float(confidence_level),
        "n": n,
        "minimum_n": int(minimum_n),
        "gains_candidate_only": gains,
        "losses_comparator_only": losses,
        "paired_difference": diff,
        "lower_confidence_bound": float(lower),
        "margin": float(margin),
        "sufficient_sample_for_declared_plan": bool(enough_n),
        "noninferiority_pass": bool(enough_n and lower >= -float(margin)),
    }
