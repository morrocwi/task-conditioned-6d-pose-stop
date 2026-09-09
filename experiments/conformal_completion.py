#!/usr/bin/env python3
"""Raw-proxy trajectory-calibrated completion-envelope ablation.

Evidence boundary: generated point clouds + numerical ICP/Kabsch only. The
online gate never receives hidden ground truth. This ablation is intentionally
kept as a negative control for the learned-shape certificate.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqts.safety import fail_closed_task_pass, safe_split_conformal_quantile
from experiments import numerical_icp as ni

CAL_SEED = 2026090921
TEST_SEEDS = (2026090931, 2026090932, 2026090933)
STRESS_SEED = 2026090991
ALPHA = 0.10
FLOOR = np.array([
    0.00025, 0.00025, 0.00025,
    math.radians(0.25), math.radians(0.25), math.radians(0.25),
], dtype=float)


def stage_true_error(ep, stage):
    return np.abs(ni.pose_error(stage.R, stage.t, ep.Rg, ep.tg))


def episode_nonconformity(ep, floor=FLOOR):
    floor = np.asarray(floor, float)
    if floor.shape != (6,) or np.any(~np.isfinite(floor)) or np.any(floor <= 0):
        raise ValueError("floor must contain six finite positive values")
    worst = 0.0
    for stage in ep.stages:
        proxy = np.asarray(stage.proxy, float)
        if proxy.shape != (6,) or np.any(~np.isfinite(proxy)) or np.any(proxy < 0):
            raise ValueError("proxy must contain six finite nonnegative values")
        ratio = stage_true_error(ep, stage) / (proxy + floor)
        if np.any(~np.isfinite(ratio)):
            raise ValueError("nonconformity ratio is non-finite")
        worst = max(worst, float(np.max(ratio)))
    return worst


def split_conformal_quantile(scores, alpha=ALPHA):
    return safe_split_conformal_quantile(scores, alpha)


def calibrate_envelope(episodes, alpha=ALPHA):
    scores = [episode_nonconformity(ep) for ep in episodes]
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


def completion_bounds(stage, q, floor=FLOOR):
    floor = np.asarray(floor, float)
    proxy = np.asarray(stage.proxy, float)
    if floor.shape != (6,) or np.any(~np.isfinite(floor)) or np.any(floor <= 0):
        return np.full(6, math.inf)
    if proxy.shape != (6,) or np.any(~np.isfinite(proxy)) or np.any(proxy < 0):
        return np.full(6, math.inf)
    q = float(q)
    if q == math.inf:
        return np.full(6, math.inf)
    if not math.isfinite(q) or q < 0:
        return np.full(6, math.inf)
    b = q * (proxy + floor)
    if np.any(~np.isfinite(b)) or np.any(b < 0):
        return np.full(6, math.inf)
    return b


def robust_task_pass(task, bounds):
    return fail_closed_task_pass(ni.TASKS[task], bounds)


def first_completion_stage(stages, task, q):
    gate_s = 0.0
    for stage in stages:
        tick = time.perf_counter()
        ok = robust_task_pass(task, completion_bounds(stage, q))
        gate_s += time.perf_counter() - tick
        if ok:
            return stage.k, True, gate_s
    return None, False, gate_s


def episode_covered(ep, q):
    for stage in ep.stages:
        if np.any(stage_true_error(ep, stage) > completion_bounds(stage, q) + 1e-15):
            return False
    return True


def eval_task_stop(ep, task, q):
    cert_k, act, gate_s = first_completion_stage(ep.stages, task, q)
    endpoint_k = cert_k if act else ep.stages[-1].k
    stage = ep.stages[endpoint_k]
    err = ni.pose_error(stage.R, stage.t, ep.Rg, ep.tg)
    success = bool(act and ni.task_success(task, err))
    return {
        "certificate_k": int(cert_k) if cert_k is not None else None,
        "endpoint_k": int(endpoint_k),
        "act": bool(act),
        "hold": bool(not act),
        "success": success,
        "unsafe_act": bool(act and not success),
        "trajectory_prefix_s": ni.cumulative_time(ep.stages, endpoint_k) + gate_s,
    }


def eval_estimator_stop(ep, task):
    k = ni.estimator_stop_stage(ep.stages)
    stage = ep.stages[k]
    err = ni.pose_error(stage.R, stage.t, ep.Rg, ep.tg)
    success = bool(ni.task_success(task, err))
    return {
        "k": int(k), "success": success,
        "trajectory_prefix_s": ni.cumulative_time(ep.stages, k),
    }


def summarize_task(episodes, task, q):
    task_rows = [eval_task_stop(ep, task, q) for ep in episodes]
    est_rows = [eval_estimator_stop(ep, task) for ep in episodes]
    n = len(episodes)
    endpoint = np.array([r["endpoint_k"] for r in task_rows], float)
    ke = np.array([r["k"] for r in est_rows], float)
    cert = [r["certificate_k"] for r in task_rows]
    act = sum(r["act"] for r in task_rows)
    succ = sum(r["success"] for r in task_rows)
    unsafe = sum(r["unsafe_act"] for r in task_rows)
    est_succ = sum(r["success"] for r in est_rows)
    before = sum(c is not None and c < e["k"] for c, e in zip(cert, est_rows))
    return {
        "episodes": n,
        "certificate": {
            "rate": act / n,
            "mean_k_given_certificate": float(np.mean([c for c in cert if c is not None])) if act else None,
            "before_estimator_rate_all_episodes": before / n,
        },
        "task_stop": {
            "mean_endpoint_k": float(np.mean(endpoint)),
            "act_rate": act / n,
            "hold_rate": 1.0 - act / n,
            "completion_rate": succ / n,
            "unsafe_act_rate": unsafe / n,
            "success_given_act": succ / act if act else None,
            "mean_trajectory_prefix_ms": float(1000.0 * np.mean([r["trajectory_prefix_s"] for r in task_rows])),
        },
        "estimator_stop": {
            "mean_k": float(np.mean(ke)),
            "completion_rate": est_succ / n,
            "unsafe_act_rate": 1.0 - est_succ / n,
            "mean_trajectory_prefix_ms": float(1000.0 * np.mean([r["trajectory_prefix_s"] for r in est_rows])),
        },
        "mean_endpoint_k_task_minus_estimator": float(np.mean(endpoint - ke)),
        "completion_difference_task_minus_estimator": float((succ - est_succ) / n),
    }


def evaluate(episodes, q):
    covered = sum(episode_covered(ep, q) for ep in episodes)
    return {
        "episodes": len(episodes),
        "trajectory_envelope_coverage": covered / len(episodes),
        "trajectory_envelope_coverage_wilson95": ni.wilson(covered, len(episodes)),
        "tasks": {task: summarize_task(episodes, task, q) for task in ni.TASKS},
    }


def generate(seed, n, max_iter, points, noise_scale=1.0):
    rng = np.random.default_rng(seed)
    return [ni.generate_episode(rng, max_iter, points, noise_scale=noise_scale) for _ in range(n)]


def render_md(payload, path):
    c = payload["calibration"]
    q_text = "+inf (uninformative/HOLD)" if c["calibration_unbounded"] else f"{c['q']:.6g}"
    lines = [
        "# Raw-proxy trajectory-calibrated completion-envelope ablation", "",
        "> Generated point clouds and ICP/Kabsch only; no camera/GPU/robot.", "",
        f"alpha={c['alpha']:.3f}; target marginal episode coverage={100*c['target_marginal_episode_coverage']:.1f}%; n={c['episodes']}; q={q_text}.", "",
        "Trajectory-prefix timing is a counterfactual cost estimate from a fully generated trajectory, not a measured online-policy speedup.", "",
    ]
    for condition in ("in_distribution", "stress_2x_sensor_noise"):
        d = payload["evaluation"][condition]
        ci = d["trajectory_envelope_coverage_wilson95"]
        lines += [f"## {condition}", "", f"Coverage: **{100*d['trajectory_envelope_coverage']:.2f}%** (Wilson 95% CI {100*ci[0]:.2f}%–{100*ci[1]:.2f}%).", ""]
        lines += ["| Task | estimator k | certificate rate | certificate k (given hit) | completion | HOLD | unsafe ACT |", "|---|---:|---:|---:|---:|---:|---:|"]
        for task, s in d["tasks"].items():
            t, e, c0 = s["task_stop"], s["estimator_stop"], s["certificate"]
            ck = "—" if c0["mean_k_given_certificate"] is None else f"{c0['mean_k_given_certificate']:.3f}"
            lines.append(f"| {task} | {e['mean_k']:.3f} | {100*c0['rate']:.2f}% | {ck} | {100*t['completion_rate']:.2f}% | {100*t['hold_rate']:.2f}% | {100*t['unsafe_act_rate']:.2f}% |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(profile, out_dir):
    if profile == "ci":
        cfg = dict(cal_n=28, test_per_seed=6, stress_n=12, max_iter=12, points=56)
    elif profile == "quick":
        cfg = dict(cal_n=64, test_per_seed=14, stress_n=24, max_iter=16, points=80)
    else:
        cfg = dict(cal_n=160, test_per_seed=40, stress_n=60, max_iter=20, points=120)

    cal = generate(CAL_SEED, cfg["cal_n"], cfg["max_iter"], cfg["points"])
    calibration, q = calibrate_envelope(cal, ALPHA)
    test = []
    for seed in TEST_SEEDS:
        test.extend(generate(seed, cfg["test_per_seed"], cfg["max_iter"], cfg["points"]))
    stress = generate(STRESS_SEED, cfg["stress_n"], cfg["max_iter"], cfg["points"], noise_scale=2.0)

    payload = {
        "schema_version": 2,
        "label": "[NumericalBackend+SplitCalibration]",
        "method": "trajectory_level_split_conformal_completion_envelope",
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
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "configuration": {**cfg, "alpha": ALPHA, "floor": FLOOR.tolist(), "cal_seed": CAL_SEED, "test_seeds": list(TEST_SEEDS), "stress_seed": STRESS_SEED},
        "calibration": calibration,
        "evaluation": {"in_distribution": evaluate(test, q), "stress_2x_sensor_noise": evaluate(stress, q)},
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "conformal_completion.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    render_md(payload, out / "CONFORMAL_COMPLETION_RESULTS.md")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("ci", "quick", "standard"), default="standard")
    ap.add_argument("--out-dir", default="results/conformal")
    args = ap.parse_args()
    payload = run(args.profile, args.out_dir)
    print(json.dumps({"calibration": payload["calibration"], "evaluation": payload["evaluation"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
