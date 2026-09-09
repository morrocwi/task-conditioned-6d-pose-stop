#!/usr/bin/env python3
"""Backend-agnostic university lab harness for coverage-qualified task stopping.

Online certificate logic sees only stages[*].features plus frozen calibration.
Oracle pose errors are used only for TRAIN/CALIBRATION/FINAL TEST evaluation.

Units: pose error = [tx,ty,tz,rx,ry,rz], metres and radians.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqts.safety import (
    CertificateNumericsError,
    fail_closed_task_pass,
    paired_binary_noninferiority,
    safe_exp_error_bound,
    safe_split_conformal_quantile,
    validate_task_spec,
)


@dataclass(frozen=True)
class Model:
    beta: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    error_floor: np.ndarray


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{ln}: invalid JSON: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no episodes")
    return rows


def validate_episode(ep, feature_dim=None):
    eid = ep.get("episode_id")
    if not isinstance(eid, str) or not eid:
        raise ValueError("episode_id must be a non-empty string")
    stages = ep.get("stages")
    errors = ep.get("oracle", {}).get("abs_pose_error_6d")
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"{eid}: stages missing")
    if not isinstance(errors, list) or len(errors) != len(stages):
        raise ValueError(f"{eid}: oracle.abs_pose_error_6d must match stages")
    ks, stops = [], 0
    for j, (stage, err) in enumerate(zip(stages, errors)):
        k, feat = stage.get("k"), stage.get("features")
        if not isinstance(k, int):
            raise ValueError(f"{eid}: stage {j} k must be int")
        ks.append(k)
        if not isinstance(feat, list) or not feat:
            raise ValueError(f"{eid}: stage {j} features missing")
        if feature_dim is not None and len(feat) != feature_dim:
            raise ValueError(f"{eid}: feature dim mismatch")
        if not all(math.isfinite(float(x)) for x in feat):
            raise ValueError(f"{eid}: non-finite feature")
        if not isinstance(err, list) or len(err) != 6 or any(float(x) < 0 or not math.isfinite(float(x)) for x in err):
            raise ValueError(f"{eid}: oracle error must be six finite nonnegative values")
        inc = float(stage.get("incremental_ms", 0.0))
        if inc < 0 or not math.isfinite(inc):
            raise ValueError(f"{eid}: invalid incremental_ms")
        stops += bool(stage.get("estimator_stop", False))
    if ks != sorted(ks) or len(set(ks)) != len(ks):
        raise ValueError(f"{eid}: stage k must be unique and sorted")
    if stops > 1:
        raise ValueError(f"{eid}: at most one estimator_stop=true")
    return len(stages[0]["features"])


def validate_dataset(rows):
    dim, ids = None, set()
    for ep in rows:
        if ep.get("episode_id") in ids:
            raise ValueError(f"duplicate episode_id {ep.get('episode_id')}")
        ids.add(ep.get("episode_id"))
        dim = validate_episode(ep, dim)
    return dim, ids


def ensure_disjoint(*datasets):
    seen = set()
    for name, rows in datasets:
        _, ids = validate_dataset(rows)
        dup = seen & ids
        if dup:
            raise ValueError(f"split leakage: {name} shares episode ids: {sorted(dup)[:5]}")
        seen |= ids


def design(feat, mean, scale):
    x = (np.asarray(feat, float) - mean) / scale
    if not np.all(np.isfinite(x)):
        raise CertificateNumericsError("non-finite standardized online feature")
    return np.r_[1.0, x]


def fit_model(train, error_floor):
    floors = np.asarray(error_floor, float)
    if floors.shape != (6,) or np.any(~np.isfinite(floors)) or np.any(floors <= 0):
        raise ValueError("error_floor must contain six finite positive values")
    feats = [np.asarray(s["features"], float) for ep in train for s in ep["stages"]]
    X0 = np.asarray(feats, float)
    mean, scale = X0.mean(0), X0.std(0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    X = np.vstack([design(f, mean, scale) for f in feats])
    betas = []
    for axis in range(6):
        y = []
        for ep in train:
            for err in ep["oracle"]["abs_pose_error_6d"]:
                y.append(math.log(float(err[axis]) + float(floors[axis])))
        beta, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
        betas.append(beta)
    beta = np.asarray(betas, float)
    if np.any(~np.isfinite(beta)):
        raise CertificateNumericsError("fitted shape model contains non-finite coefficients")
    return Model(beta, mean, scale, floors)


def predicted_log_error(stage, axis, model):
    value = float(design(stage["features"], model.mean, model.scale) @ model.beta[axis])
    if not math.isfinite(value):
        raise CertificateNumericsError("predicted log error is non-finite")
    return value


def nonconformity(ep, model):
    worst = -math.inf
    for stage, err in zip(ep["stages"], ep["oracle"]["abs_pose_error_6d"]):
        for axis in range(6):
            value = math.log(float(err[axis]) + float(model.error_floor[axis])) - predicted_log_error(stage, axis, model)
            if not math.isfinite(value):
                raise CertificateNumericsError("nonconformity score is non-finite")
            worst = max(worst, value)
    if not math.isfinite(worst):
        raise CertificateNumericsError("episode nonconformity is invalid")
    return float(worst)


def conformal_quantile(scores, alpha):
    return safe_split_conformal_quantile(scores, alpha)


def completion_bounds(stage, model, q):
    try:
        out = [
            safe_exp_error_bound(predicted_log_error(stage, axis, model), q, model.error_floor[axis])
            for axis in range(6)
        ]
        return np.asarray(out, float)
    except (CertificateNumericsError, ValueError, FloatingPointError, OverflowError):
        return np.full(6, math.inf, dtype=float)


def task_pass(spec, x):
    return fail_closed_task_pass(spec, np.abs(np.asarray(x, float)))


def first_certificate_stage(stages, spec, model, q):
    """Return certificate index or None. Ground truth is absent from the signature."""
    for idx, stage in enumerate(stages):
        if task_pass(spec, completion_bounds(stage, model, q)):
            return idx, True
    return None, False


def estimator_index(stages):
    for i, stage in enumerate(stages):
        if bool(stage.get("estimator_stop", False)):
            return i
    return len(stages) - 1


def cumulative_ms(stages, idx):
    return float(sum(float(s.get("incremental_ms", 0.0)) for s in stages[: idx + 1]))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def boot_mean_diff(a, b, seed=12345, repeats=5000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d, n = a - b, len(a)
    rng, vals = np.random.default_rng(seed), np.empty(repeats)
    for i in range(repeats):
        idx = rng.integers(0, n, size=n)
        vals[i] = d[idx].mean()
    return {"mean": float(d.mean()), "bootstrap95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]}


def _measured_online_pair(ep, task_name):
    rec = ep.get("policy_timing_ms", {}).get(task_name)
    if not isinstance(rec, dict):
        raise ValueError(f"{ep['episode_id']}: missing policy_timing_ms.{task_name}")
    c = float(rec.get("certificate_stop"))
    e = float(rec.get("estimator_stop"))
    if c < 0 or e < 0 or not math.isfinite(c) or not math.isfinite(e):
        raise ValueError(f"{ep['episode_id']}: invalid measured policy timing")
    return c, e


def evaluate(test, tasks, model, q, inference, repeats, timing_mode):
    covered = [nonconformity(ep, model) <= q + 1e-12 for ep in test]
    n, cov_n = len(test), sum(covered)
    result = {
        "episodes": n,
        "trajectory_envelope_coverage": cov_n / n,
        "trajectory_envelope_coverage_wilson95": wilson(cov_n, n),
        "tasks": {},
    }
    for ti, (name, spec) in enumerate(tasks.items()):
        validate_task_spec(spec)
        cert_k, endpoint_k, est_k = [], [], []
        c_succ, e_succ, hold, unsafe = [], [], [], []
        prefix_c, prefix_e, measured_c, measured_e = [], [], [], []
        unsafe_cov = 0
        for ep, cov in zip(test, covered):
            ci, act = first_certificate_stage(ep["stages"], spec, model, q)
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
            if timing_mode == "online_policy_measured":
                cm, em = _measured_online_pair(ep, name)
                measured_c.append(cm)
                measured_e.append(em)

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
        if timing_mode == "online_policy_measured":
            measured_diff = boot_mean_diff(measured_c, measured_e, 4000 + ti, repeats)
            task_out["timing"] = {
                "mode": timing_mode,
                "prefix_cost_is_counterfactual": False,
                "paired_online_policy_ms_difference": measured_diff,
                "latency_reduction_pass": bool(measured_diff["bootstrap95"][1] < 0.0),
            }
        if unsafe_cov:
            raise AssertionError(f"{name}: unsafe ACT occurred on an episode covered by the calibrated envelope")
        result["tasks"][name] = task_out
    return result


def _load_and_validate_splits(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train, cal, test = load_jsonl(args.train), load_jsonl(args.calibration), load_jsonl(args.test)
    ensure_disjoint(("train", train), ("calibration", cal), ("test", test))
    d1, _ = validate_dataset(train)
    d2, _ = validate_dataset(cal)
    d3, _ = validate_dataset(test)
    if len({d1, d2, d3}) != 1:
        raise ValueError("feature dimension differs across splits")
    return cfg, train, cal, test, d1


def _common_config(cfg):
    alpha = float(cfg.get("alpha", 0.1))
    floors = cfg.get("error_floor", [1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4])
    inference_cfg = cfg.get("inference", {})
    inference = {
        "margin": float(inference_cfg.get("noninferiority_margin", cfg.get("noninferiority_margin", 0.05))),
        "confidence_level": float(inference_cfg.get("confidence_level", 0.95)),
        "minimum_test_episodes": int(inference_cfg.get("minimum_test_episodes", 30)),
        "independent_sampling_unit": str(inference_cfg.get("independent_sampling_unit", "episode")),
    }
    if not inference["independent_sampling_unit"]:
        raise ValueError("independent_sampling_unit must be predeclared")
    timing_mode = str(cfg.get("timing_mode", "trajectory_prefix_estimate"))
    if timing_mode not in {"trajectory_prefix_estimate", "online_policy_measured"}:
        raise ValueError("timing_mode must be trajectory_prefix_estimate or online_policy_measured")
    for spec in cfg["tasks"].values():
        validate_task_spec(spec)
    return alpha, floors, inference, timing_mode


def run_whole_trajectory(args):
    cfg, train, cal, test, d1 = _load_and_validate_splits(args)
    alpha, floors, inference, timing_mode = _common_config(cfg)

    model = fit_model(train, floors)
    scores = [nonconformity(ep, model) for ep in cal]
    q, rank = conformal_quantile(scores, alpha)
    ev = evaluate(test, cfg["tasks"], model, q, inference, int(cfg.get("bootstrap_repeats", 5000)), timing_mode)

    payload = {
        "schema_version": 2,
        "claim": "coverage-qualified downstream task stopping of iterative 6D pose refinement",
        "units": {"translation": "metres", "rotation": "radians", "time": "milliseconds"},
        "protocol": {
            "mode": "whole_trajectory",
            "train_episodes": len(train),
            "calibration_episodes": len(cal),
            "test_episodes": len(test),
            "alpha": alpha,
            "target_marginal_whole_trajectory_coverage": 1 - alpha,
            "calibration_quantile_rank": rank,
            "calibration_unbounded": bool(q == math.inf),
            "q": None if q == math.inf else q,
            "feature_dim": d1,
            "inference": inference,
            "timing_mode": timing_mode,
        },
        "evaluation": ev,
        "evidence_boundary": {
            "online_gate_uses_oracle": False,
            "physical_robot_result_inferred_from_pose_error": False,
            "trajectory_prefix_timing_is_not_online_speedup": timing_mode != "online_policy_measured",
            "result_scope": "instrumented iterative 6D pose backend; task readers are declared pose-error admissibility tests",
        },
    }
    return payload


def run_bonferroni_multicheckpoint(args):
    # Imported here (not at module top) to avoid a hard import-time dependency
    # from the existing whole-trajectory path onto the new module, and to
    # avoid a circular import (lab.multicheckpoint imports several functions
    # back from this module).
    from lab.multicheckpoint import (
        calibrate_checkpoints,
        evaluate_multicheckpoint,
        validate_checkpoints,
    )

    cfg, train, cal, test, d1 = _load_and_validate_splits(args)
    alpha, floors, inference, timing_mode = _common_config(cfg)
    checkpoints = validate_checkpoints(cfg.get("checkpoints"), train[0]["stages"])

    model = fit_model(train, floors)
    q_by_checkpoint = calibrate_checkpoints(cal, model, checkpoints, alpha)
    ev = evaluate_multicheckpoint(
        test, cfg["tasks"], model, checkpoints, q_by_checkpoint, inference,
        int(cfg.get("bootstrap_repeats", 5000)), timing_mode,
    )

    payload = {
        "schema_version": 2,
        "claim": "Bonferroni-corrected multi-checkpoint coverage-qualified downstream task stopping of iterative 6D pose refinement (PROP-CONF-03)",
        "units": {"translation": "metres", "rotation": "radians", "time": "milliseconds"},
        "protocol": {
            "mode": "bonferroni_multicheckpoint",
            "train_episodes": len(train),
            "calibration_episodes": len(cal),
            "test_episodes": len(test),
            "alpha": alpha,
            "checkpoints": [int(k) for k in checkpoints],
            "num_checkpoints": len(checkpoints),
            "per_checkpoint_alpha": alpha / len(checkpoints),
            "target_marginal_whole_trajectory_coverage": 1 - alpha,
            "feature_dim": d1,
            "inference": inference,
            "timing_mode": timing_mode,
            "reference": "PROP-CONF-03, ~/ANSE.ASIA/toledo/registry/proposals/conformal_stopping_family.json; union bound machine-checked in ~/ANSE.ASIA/toledo/coq/canonical/PROP_CONF_03_union_bound.v",
        },
        "evaluation": ev,
        "evidence_boundary": {
            "online_gate_uses_oracle": False,
            "physical_robot_result_inferred_from_pose_error": False,
            "trajectory_prefix_timing_is_not_online_speedup": timing_mode != "online_policy_measured",
            "result_scope": "instrumented iterative 6D pose backend; task readers are declared pose-error admissibility tests; certificate is checked ONLY at the predeclared checkpoint stages, each against its own Bonferroni-corrected quantile",
        },
    }
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="lab_results.json")
    ap.add_argument(
        "--mode",
        default="whole_trajectory",
        choices=["whole_trajectory", "bonferroni_multicheckpoint"],
        help=(
            "whole_trajectory: existing C7-C9 joint max-over-stages construction "
            "(runs 1-2, unchanged). bonferroni_multicheckpoint: new PROP-CONF-03 "
            "per-checkpoint construction (this run)."
        ),
    )
    args = ap.parse_args()

    if args.mode == "bonferroni_multicheckpoint":
        payload = run_bonferroni_multicheckpoint(args)
    else:
        payload = run_whole_trajectory(args)

    Path(args.out).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
