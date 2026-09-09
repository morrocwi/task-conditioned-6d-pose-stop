#!/usr/bin/env python3
"""Learned-shape, trajectory-conformal completion envelopes.

TRAIN fits an observable error-shape model; CALIBRATION conformalizes one score
per complete refinement trajectory; FINAL TEST evaluates the frozen certificate.
The online gate receives no hidden ground-truth pose.

Evidence boundary: generated point clouds + numerical ICP/Kabsch only.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqts.safety import (
    CertificateNumericsError,
    fail_closed_task_pass,
    safe_exp_error_bound,
    safe_split_conformal_quantile,
)
from experiments import numerical_icp as ni

TRAIN_SEED = 2026090920
CAL_SEED = 2026090951
FINAL_TEST_SEEDS = (2026090961, 2026090962, 2026090963)
STRESS_SEED = 2026090993
ALPHA = 0.10

ERROR_FLOOR = np.array([1e-5, 1e-5, 1e-5, math.radians(0.01), math.radians(0.01), math.radians(0.01)], dtype=float)
PROXY_FLOOR = np.array([5e-5, 5e-5, 5e-5, math.radians(0.05), math.radians(0.05), math.radians(0.05)], dtype=float)
DELTA_FLOOR = np.array([5e-5, 5e-5, 5e-5, math.radians(0.02), math.radians(0.02), math.radians(0.02)], dtype=float)
RMS_FLOOR = 1e-5


@dataclass(frozen=True)
class ShapeModel:
    beta: np.ndarray
    max_iter: int


def hidden_error(ep, stage):
    return np.abs(ni.pose_error(stage.R, stage.t, ep.Rg, ep.tg))


def observable_features(stage, axis, max_iter):
    vals = np.array([
        1.0,
        math.log(float(stage.proxy[axis]) + float(PROXY_FLOOR[axis])),
        math.log(abs(float(stage.proposed_delta[axis])) + float(DELTA_FLOOR[axis])),
        math.log(float(stage.rms) + RMS_FLOOR),
        float(stage.k) / max(1.0, float(max_iter)),
    ], dtype=float)
    if np.any(~np.isfinite(vals)):
        raise CertificateNumericsError("non-finite observable feature")
    return vals


def fit_shape_model(episodes, max_iter):
    betas = []
    for axis in range(6):
        X, y = [], []
        for ep in episodes:
            for stage in ep.stages:
                X.append(observable_features(stage, axis, max_iter))
                err = float(hidden_error(ep, stage)[axis])
                y.append(math.log(err + float(ERROR_FLOOR[axis])))
        beta, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)
        betas.append(beta)
    beta = np.asarray(betas, float)
    if beta.shape != (6, 5) or np.any(~np.isfinite(beta)):
        raise CertificateNumericsError("shape model coefficients are invalid")
    return ShapeModel(beta=beta, max_iter=max_iter)


def predicted_log_error(stage, axis, model):
    if np.any(~np.isfinite(model.beta)):
        raise CertificateNumericsError("shape model contains non-finite coefficients")
    value = float(observable_features(stage, axis, model.max_iter) @ model.beta[axis])
    if not math.isfinite(value):
        raise CertificateNumericsError("predicted log error is non-finite")
    return value


def episode_nonconformity(ep, model):
    worst = -math.inf
    for stage in ep.stages:
        err = hidden_error(ep, stage)
        for axis in range(6):
            residual = math.log(float(err[axis]) + float(ERROR_FLOOR[axis])) - predicted_log_error(stage, axis, model)
            if not math.isfinite(residual):
                raise CertificateNumericsError("nonconformity residual is non-finite")
            worst = max(worst, residual)
    if not math.isfinite(worst):
        raise CertificateNumericsError("episode nonconformity is invalid")
    return float(worst)


def split_conformal_quantile(scores, alpha=ALPHA):
    return safe_split_conformal_quantile(scores, alpha)


def calibrate(episodes, model, alpha=ALPHA):
    scores = [episode_nonconformity(ep, model) for ep in episodes]
    q, rank = split_conformal_quantile(scores, alpha)
    return {
        "alpha": float(alpha),
        "target_marginal_episode_coverage": float(1.0 - alpha),
        "episodes": len(episodes),
        "quantile_rank": rank,
        "calibration_unbounded": bool(q == math.inf),
        "q": None if q == math.inf else q,
        "score_min": float(np.min(scores)),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
    }, q


def completion_bounds(stage, model, q):
    try:
        out = [safe_exp_error_bound(predicted_log_error(stage, axis, model), q, ERROR_FLOOR[axis]) for axis in range(6)]
        return np.asarray(out, float)
    except (CertificateNumericsError, ValueError, FloatingPointError, OverflowError):
        return np.full(6, math.inf, dtype=float)


def robust_task_pass(task, bounds):
    return fail_closed_task_pass(ni.TASKS[task], bounds)


def first_completion_stage(stages, task, model, q):
    """Return certificate hitting stage or None; ground truth is absent."""
    gate_s = 0.0
    for stage in stages:
        tick = time.perf_counter()
        passed = robust_task_pass(task, completion_bounds(stage, model, q))
        gate_s += time.perf_counter() - tick
        if passed:
            return stage.k, True, gate_s
    return None, False, gate_s


def episode_covered(ep, model, q):
    if q == math.inf:
        return True
    return bool(episode_nonconformity(ep, model) <= float(q) + 1e-12)


def estimator_result(ep, task):
    k = ni.estimator_stop_stage(ep.stages)
    stage = ep.stages[k]
    ok = bool(ni.task_success(task, ni.pose_error(stage.R, stage.t, ep.Rg, ep.tg)))
    return {"k": int(k), "success": ok, "trajectory_prefix_s": ni.cumulative_time(ep.stages, k)}


def completion_result(ep, task, model, q):
    cert_k, act, gate_s = first_completion_stage(ep.stages, task, model, q)
    endpoint_k = cert_k if act else ep.stages[-1].k
    stage = ep.stages[endpoint_k]
    ok = bool(act and ni.task_success(task, ni.pose_error(stage.R, stage.t, ep.Rg, ep.tg)))
    return {
        "certificate_k": int(cert_k) if cert_k is not None else None,
        "endpoint_k": int(endpoint_k),
        "act": bool(act),
        "hold": bool(not act),
        "success": ok,
        "unsafe_act": bool(act and not ok),
        "trajectory_prefix_s": ni.cumulative_time(ep.stages, endpoint_k) + gate_s,
    }


def summarize_task(episodes, task, model, q):
    covered = [episode_covered(ep, model, q) for ep in episodes]
    crows = [completion_result(ep, task, model, q) for ep in episodes]
    erows = [estimator_result(ep, task) for ep in episodes]
    n = len(episodes)
    endpoints = np.asarray([r["endpoint_k"] for r in crows], float)
    ke = np.asarray([r["k"] for r in erows], float)
    cert = [r["certificate_k"] for r in crows]
    act = sum(r["act"] for r in crows)
    success = sum(r["success"] for r in crows)
    unsafe = sum(r["unsafe_act"] for r in crows)
    unsafe_while_covered = sum(bool(r["unsafe_act"] and cov) for r, cov in zip(crows, covered))
    est_success = sum(r["success"] for r in erows)
    before = sum(c is not None and c < e["k"] for c, e in zip(cert, erows))
    return {
        "episodes": n,
        "certificate": {
            "rate": act / n,
            "mean_k_given_certificate": float(np.mean([c for c in cert if c is not None])) if act else None,
            "before_estimator_rate_all_episodes": before / n,
            "before_estimator_rate_given_certificate": before / act if act else None,
        },
        "completion_stop": {
            "mean_endpoint_k": float(np.mean(endpoints)),
            "act_rate": act / n,
            "hold_rate": 1.0 - act / n,
            "completion_rate": success / n,
            "unsafe_act_rate": unsafe / n,
            "success_given_act": success / act if act else None,
            "mean_trajectory_prefix_ms": float(1000.0 * np.mean([r["trajectory_prefix_s"] for r in crows])),
        },
        "estimator_stop": {
            "mean_k": float(np.mean(ke)),
            "completion_rate": est_success / n,
            "unsafe_act_rate": 1.0 - est_success / n,
            "mean_trajectory_prefix_ms": float(1000.0 * np.mean([r["trajectory_prefix_s"] for r in erows])),
        },
        "mean_endpoint_k_completion_minus_estimator": float(np.mean(endpoints - ke)),
        "completion_difference_completion_minus_estimator": float((success - est_success) / n),
        "unsafe_act_on_covered_episode_count": int(unsafe_while_covered),
    }


def evaluate(episodes, model, q):
    coverage_flags = [episode_covered(ep, model, q) for ep in episodes]
    covered = sum(coverage_flags)
    tasks = {task: summarize_task(episodes, task, model, q) for task in ni.TASKS}
    if any(v["unsafe_act_on_covered_episode_count"] != 0 for v in tasks.values()):
        raise AssertionError("robust task gate allowed unsafe ACT inside a covered envelope")
    return {
        "episodes": len(episodes),
        "trajectory_envelope_coverage": covered / len(episodes),
        "trajectory_envelope_coverage_wilson95": ni.wilson(covered, len(episodes)),
        "tasks": tasks,
    }


def generate(seed, n, max_iter, points, noise_scale=1.0):
    rng = np.random.default_rng(seed)
    return [ni.generate_episode(rng, max_iter, points, noise_scale=noise_scale) for _ in range(n)]


def render_md(payload, path):
    c = payload["calibration"]
    q_text = "+inf (uninformative/HOLD)" if c["calibration_unbounded"] else f"{c['q']:.6g}"
    lines = [
        "# Learned-shape trajectory-conformal completion results", "",
        "> Generated point clouds and ICP/Kabsch only; no camera/GPU/robot.", "",
        f"Target marginal whole-trajectory coverage: {100*c['target_marginal_episode_coverage']:.1f}% (alpha={c['alpha']:.3f}); calibration episodes={c['episodes']}; q={q_text}.", "",
        "Trajectory-prefix timing is a counterfactual prefix-cost estimate from a fully generated trajectory. It is not a measured online-policy speedup.", "",
    ]
    for condition in ("in_distribution", "stress_2x_sensor_noise"):
        d = payload["evaluation"][condition]
        ci = d["trajectory_envelope_coverage_wilson95"]
        lines += [f"## {condition}", "", f"Whole-trajectory envelope coverage: **{100*d['trajectory_envelope_coverage']:.2f}%** (Wilson 95% CI {100*ci[0]:.2f}%–{100*ci[1]:.2f}%).", ""]
        lines += ["| Task | estimator mean k | certificate rate | certificate k (given hit) | cert<est all eps | estimator completion | certificate-policy completion | HOLD | unsafe ACT |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for task, s in d["tasks"].items():
            a, e, cert = s["completion_stop"], s["estimator_stop"], s["certificate"]
            ck = "—" if cert["mean_k_given_certificate"] is None else f"{cert['mean_k_given_certificate']:.3f}"
            lines.append(f"| {task} | {e['mean_k']:.3f} | {100*cert['rate']:.2f}% | {ck} | {100*cert['before_estimator_rate_all_episodes']:.2f}% | {100*e['completion_rate']:.2f}% | {100*a['completion_rate']:.2f}% | {100*a['hold_rate']:.2f}% | {100*a['unsafe_act_rate']:.2f}% |")
        lines.append("")
    lines += [
        "## Mechanical implication check", "",
        "CI verifies zero unsafe ACTs on episodes whose true numerical trajectory is contained by the calibrated envelope under the matching declared reader. This is an implementation consistency check, not a physical safety certificate.", "",
        "## Evidence boundary", "",
        "Split-conformal coverage is marginal under exchangeability. The 2x-noise stress condition is shifted and receives no transferred guarantee. Real RGB-D/GPU latency, physical manipulation, and reader validity remain open.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(profile, out_dir):
    if profile == "ci":
        cfg = dict(train_n=36, cal_n=36, test_per_seed=6, stress_n=12, max_iter=12, points=56)
    elif profile == "quick":
        cfg = dict(train_n=72, cal_n=72, test_per_seed=14, stress_n=24, max_iter=16, points=80)
    else:
        cfg = dict(train_n=160, cal_n=160, test_per_seed=40, stress_n=60, max_iter=20, points=120)

    train = generate(TRAIN_SEED, cfg["train_n"], cfg["max_iter"], cfg["points"])
    model = fit_shape_model(train, cfg["max_iter"])
    cal = generate(CAL_SEED, cfg["cal_n"], cfg["max_iter"], cfg["points"])
    calibration, q = calibrate(cal, model, ALPHA)
    final_test = []
    for seed in FINAL_TEST_SEEDS:
        final_test.extend(generate(seed, cfg["test_per_seed"], cfg["max_iter"], cfg["points"]))
    stress = generate(STRESS_SEED, cfg["stress_n"], cfg["max_iter"], cfg["points"], noise_scale=2.0)

    payload = {
        "schema_version": 2,
        "label": "[NumericalBackend+LearnedShape+SplitConformal]",
        "method": "train_shape_then_trajectory_level_split_conformal_completion_envelope",
        "evidence_boundary": {
            "generated_point_clouds": True,
            "iterative_rigid_registration_executed": True,
            "hidden_ground_truth_used_online_by_gate": False,
            "trajectory_prefix_timing_is_online_speedup": False,
            "camera_executed": False,
            "rgbd_sensor_executed": False,
            "neural_pose_model_executed": False,
            "physical_robot_executed": False,
        },
        "lineage": {
            "previous_development_test_seeds_not_reused": [2026090931, 2026090932, 2026090933],
            "train_seed": TRAIN_SEED,
            "calibration_seed": CAL_SEED,
            "final_test_seeds": list(FINAL_TEST_SEEDS),
            "stress_seed": STRESS_SEED,
        },
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "configuration": {**cfg, "alpha": ALPHA, "error_floor": ERROR_FLOOR.tolist(), "proxy_floor": PROXY_FLOOR.tolist(), "delta_floor": DELTA_FLOOR.tolist(), "rms_floor": RMS_FLOOR},
        "shape_model": {"beta": model.beta.tolist(), "features": ["intercept", "log_proxy", "log_abs_proposed_delta", "log_rms", "normalized_stage"]},
        "calibration": calibration,
        "evaluation": {"in_distribution": evaluate(final_test, model, q), "stress_2x_sensor_noise": evaluate(stress, model, q)},
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "learned_conformal_completion.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    render_md(payload, out / "LEARNED_CONFORMAL_RESULTS.md")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("ci", "quick", "standard"), default="standard")
    ap.add_argument("--out-dir", default="results/learned_conformal")
    args = ap.parse_args()
    payload = run(args.profile, args.out_dir)
    print(json.dumps({"calibration": payload["calibration"], "evaluation": payload["evaluation"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
