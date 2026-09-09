"""Bonferroni-corrected multi-checkpoint conformal certificate (PROP-CONF-03).

This is a NEW code path, additional to (never a replacement for) the existing
whole-trajectory construction in `lab/run_real_system.py` (C7-C9 / PROP-CONF-01
and PROP-CONF-02). Runs 1-2 used the whole-trajectory path and stay
reproducible: `lab/run_real_system.py` is unmodified except for an added
`--mode` flag that dispatches into this module.

Design (Toledo proposal PROP-CONF-03,
~/ANSE.ASIA/toledo/registry/proposals/conformal_stopping_family.json;
union-bound argument machine-checked, axiom-free, in
~/ANSE.ASIA/toledo/coq/canonical/PROP_CONF_03_union_bound.v; diagnosis in
~/ANSE.ASIA/glosa/projects/GLS-2026-005_pose-stop-conformal-diagnosis/
DIAGNOSIS_HYPOTHESIS.md):

  For K' predeclared checkpoint stages k_1..k_{K'} (K' <= 4 feasible at
  n=40 calibration episodes, alpha=0.1 -- see
  cqts.safety.bonferroni_feasible_max_checkpoints):

    A_{j,k_m} = max_i [ log(|e_{j,k_m,i}| + delta_i) - log(s_{j,k_m,i}) ]

  i.e. the nonconformity score at checkpoint k_m is a max over the 6 pose
  COORDINATES ONLY -- never over stages, unlike the whole-trajectory
  PROP-CONF-02 score used by runs 1-2. Each checkpoint is then calibrated
  independently at level alpha/K' by calling the SAME
  `cqts.safety.safe_split_conformal_quantile` order-statistic function used
  by the whole-trajectory path (via `cqts.safety.safe_multi_checkpoint_quantiles`,
  which is a thin per-checkpoint dispatcher over that same function -- the
  conformal quantile logic itself is never reimplemented here).

  The envelope at each checkpoint is built the same way as C9
  (`cqts.safety.safe_exp_error_bound`, reused via
  `lab.run_real_system.completion_bounds`, unchanged). The certificate fires
  at a checkpoint if that checkpoint's envelope is inside the task-admissible
  region (`cqts.safety.fail_closed_task_pass`, unchanged, via
  `lab.run_real_system.task_pass`).

Model fitting, feature standardization, and per-axis predicted-log-error
(`fit_model`, `design`, `predicted_log_error`) are imported unchanged from
`lab/run_real_system.py` -- they do not depend on the aggregation scheme, so
they are reused exactly, not duplicated.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqts.safety import (
    CertificateNumericsError,
    paired_binary_noninferiority,
    safe_multi_checkpoint_quantiles,
    validate_task_spec,
)
from lab.run_real_system import (
    boot_mean_diff,
    completion_bounds,
    cumulative_ms,
    predicted_log_error,
    task_pass,
    wilson,
)


def validate_checkpoints(checkpoints, stages_template):
    """Predeclared checkpoints must be a sorted list of distinct stage k values
    that actually occur in every episode's stage list (checked by the caller
    per-episode as it iterates; this only checks the predeclared list shape)."""
    if not isinstance(checkpoints, (list, tuple)) or not checkpoints:
        raise ValueError("checkpoints must be a non-empty list")
    ks = [int(k) for k in checkpoints]
    if ks != sorted(ks) or len(set(ks)) != len(ks):
        raise ValueError("checkpoints must be sorted, distinct stage k values")
    available = {int(s["k"]) for s in stages_template}
    missing = [k for k in ks if k not in available]
    if missing:
        raise ValueError(f"checkpoint stage(s) not present in stage template: {missing}")
    return ks


def stage_and_error_at_k(ep, k):
    """Look up the stage dict + oracle error 6-vector for stage index k in an
    episode. Fails closed (raises) rather than silently skipping a missing
    predeclared checkpoint -- a data/config mismatch must be visible, never
    treated as an implicit HOLD."""
    for stage, err in zip(ep["stages"], ep["oracle"]["abs_pose_error_6d"]):
        if int(stage["k"]) == int(k):
            return stage, err
    raise ValueError(f"{ep['episode_id']}: predeclared checkpoint k={k} not found in stages")


def stage_index_at_k(ep, k):
    for idx, stage in enumerate(ep["stages"]):
        if int(stage["k"]) == int(k):
            return idx
    raise ValueError(f"{ep['episode_id']}: predeclared checkpoint k={k} not found in stages")


def nonconformity_at_checkpoint(ep, model, k):
    """PROP-CONF-02-restricted-to-one-stage nonconformity score: max over the
    6 pose coordinates ONLY, at the single predeclared checkpoint stage k --
    never a max over stages. This is what makes the per-checkpoint calibration
    a genuinely different (and, per PROP-CONF-03, less conservative) object
    than the whole-trajectory C7 score used by runs 1-2."""
    stage, err = stage_and_error_at_k(ep, k)
    worst = -math.inf
    for axis in range(6):
        value = math.log(float(err[axis]) + float(model.error_floor[axis])) - predicted_log_error(stage, axis, model)
        if not math.isfinite(value):
            raise CertificateNumericsError("checkpoint nonconformity score is non-finite")
        worst = max(worst, value)
    if not math.isfinite(worst):
        raise CertificateNumericsError("checkpoint nonconformity is invalid")
    return float(worst)


def calibrate_checkpoints(cal, model, checkpoints, alpha):
    """Compute, for each predeclared checkpoint, the calibration-set
    nonconformity scores and the Bonferroni-corrected quantile (q, rank),
    reusing `cqts.safety.safe_multi_checkpoint_quantiles` (itself a thin
    per-checkpoint dispatcher over the unmodified
    `safe_split_conformal_quantile`)."""
    scores_by_checkpoint = {
        k: [nonconformity_at_checkpoint(ep, model, k) for ep in cal] for k in checkpoints
    }
    return safe_multi_checkpoint_quantiles(scores_by_checkpoint, alpha)


def first_certificate_checkpoint(ep, spec, model, checkpoints, q_by_checkpoint):
    """Return (stage_index, True) for the first predeclared checkpoint (in
    ascending order) whose C9-style envelope, built with that checkpoint's
    OWN Bonferroni-corrected q, lies inside the task-admissible region.
    Ground truth is absent from this signature, matching
    `run_real_system.first_certificate_stage`."""
    for k in checkpoints:
        stage, _ = stage_and_error_at_k(ep, k)
        q, _rank = q_by_checkpoint[k]
        bounds = completion_bounds(stage, model, q)
        if task_pass(spec, bounds):
            return stage_index_at_k(ep, k), True
    return None, False


def estimator_index(stages):
    for i, stage in enumerate(stages):
        if bool(stage.get("estimator_stop", False)):
            return i
    return len(stages) - 1


def evaluate_multicheckpoint(test, tasks, model, checkpoints, q_by_checkpoint, inference, repeats, timing_mode):
    """Mirrors `run_real_system.evaluate`, but the certificate is checked ONLY
    at the predeclared checkpoint stages, each against its own
    Bonferroni-corrected q, instead of at every stage against one joint q."""
    per_checkpoint_covered = {
        k: [nonconformity_at_checkpoint(ep, model, k) <= q_by_checkpoint[k][0] + 1e-12 for ep in test]
        for k in checkpoints
    }
    n = len(test)
    checkpoint_coverage = {
        int(k): {
            "coverage": sum(per_checkpoint_covered[k]) / n,
            "coverage_wilson95": wilson(sum(per_checkpoint_covered[k]), n),
            "q": None if q_by_checkpoint[k][0] == math.inf else q_by_checkpoint[k][0],
            "calibration_unbounded": bool(q_by_checkpoint[k][0] == math.inf),
            "calibration_quantile_rank": q_by_checkpoint[k][1],
        }
        for k in checkpoints
    }
    # Union-bound event: the whole-trajectory guarantee holds on an episode
    # only if EVERY checkpoint is individually covered (P(any miscoverage) <= alpha).
    all_checkpoints_covered = [
        all(per_checkpoint_covered[k][i] for k in checkpoints) for i in range(n)
    ]
    cov_n = sum(all_checkpoints_covered)

    result = {
        "checkpoints": [int(k) for k in checkpoints],
        "per_checkpoint": checkpoint_coverage,
        "all_checkpoints_covered_rate": cov_n / n,
        "all_checkpoints_covered_wilson95": wilson(cov_n, n),
        "tasks": {},
    }
    for ti, (name, spec) in enumerate(tasks.items()):
        validate_task_spec(spec)
        cert_k, endpoint_k, est_k = [], [], []
        c_succ, e_succ, hold, unsafe = [], [], [], []
        prefix_c, prefix_e = [], []
        unsafe_cov = 0
        for ep, cov in zip(test, all_checkpoints_covered):
            ci, act = first_certificate_checkpoint(ep, spec, model, checkpoints, q_by_checkpoint)
            ei = estimator_index(ep["stages"])
            end_i = ci if act else len(ep["stages"]) - 1
            cerr = np.asarray(ep["oracle"]["abs_pose_error_6d"][end_i], float)
            eerr = np.asarray(ep["oracle"]["abs_pose_error_6d"][ei], float)
            cs = bool(act and task_pass(spec, cerr))
            es = bool(task_pass(spec, eerr))
            un = bool(act and not cs)
            if un and cov:
                unsafe_cov += 1
            cert_k.append(ep["stages"][ci]["k"] if act else None)
            endpoint_k.append(ep["stages"][end_i]["k"])
            est_k.append(ep["stages"][ei]["k"])
            c_succ.append(int(cs))
            e_succ.append(int(es))
            hold.append(int(not act))
            unsafe.append(int(un))
            prefix_c.append(cumulative_ms(ep["stages"], end_i))
            prefix_e.append(cumulative_ms(ep["stages"], ei))

        endpoint_diff = boot_mean_diff(endpoint_k, est_k, 1000 + ti, repeats)
        prefix_diff = boot_mean_diff(prefix_c, prefix_e, 3000 + ti, repeats)
        ni = paired_binary_noninferiority(
            c_succ,
            e_succ,
            margin=float(inference["margin"]),
            confidence_level=float(inference["confidence_level"]),
            minimum_n=int(inference["minimum_test_episodes"]),
        )
        certified_pairs = [(c, e) for c, e in zip(cert_k, est_k) if c is not None]
        before = sum(c < e for c, e in certified_pairs)
        task_out = {
            "certificate_hitting_time": {
                "certificate_rate": float(np.mean([c is not None for c in cert_k])),
                "mean_k_given_certificate": float(np.mean([c for c in cert_k if c is not None])) if certified_pairs else None,
                "certificate_before_estimator_rate_all_episodes": before / n,
                "certificate_before_estimator_rate_given_certificate": before / len(certified_pairs) if certified_pairs else None,
            },
            "executed_certificate_policy": {
                "mean_endpoint_k": float(np.mean(endpoint_k)),
                "completion_rate": float(np.mean(c_succ)),
                "hold_rate": float(np.mean(hold)),
                "unsafe_act_rate": float(np.mean(unsafe)),
                "mean_trajectory_prefix_ms": float(np.mean(prefix_c)),
            },
            "estimator_stop": {
                "mean_k": float(np.mean(est_k)),
                "completion_rate": float(np.mean(e_succ)),
                "mean_trajectory_prefix_ms": float(np.mean(prefix_e)),
            },
            "paired_endpoint_k_difference": endpoint_diff,
            "paired_prefix_perception_ms_difference": prefix_diff,
            "noninferiority": ni,
            "unsafe_act_on_covered_episode_count": int(unsafe_cov),
            "timing": {
                "mode": timing_mode,
                "prefix_cost_is_counterfactual": True,
                "latency_reduction_pass": None,
            },
        }
        if unsafe_cov:
            raise AssertionError(f"{name}: unsafe ACT occurred on an episode covered by every predeclared checkpoint")
        result["tasks"][name] = task_out
    return result
