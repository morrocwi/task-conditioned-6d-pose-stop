#!/usr/bin/env python3
"""Learned-shape, trajectory-conformal completion envelopes.

This is the stronger numerical experiment for coverage-qualified task stopping.
It separates three roles explicitly:

1. TRAIN: fit an observable shape model for pose-error magnitude.
2. CALIBRATION: conformalize one score per complete refinement episode.
3. FINAL TEST: evaluate frozen coverage and task stopping on untouched seeds.

The online gate receives only retained refinement stages, the frozen shape model,
the frozen conformal scale, and the declared task. Ground-truth pose is absent
from the gate interface.

Evidence boundary: generated point clouds + numerical ICP/Kabsch only. No
camera, RGB-D sensor, neural pose network, contact physics, or physical robot.
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
from experiments import numerical_icp as ni

# Development lineage:
# 2026090931..33 were inspected during method development and are therefore
# NOT reused as final-test seeds here.
TRAIN_SEED = 2026090920
CAL_SEED = 2026090951
FINAL_TEST_SEEDS = (2026090961, 2026090962, 2026090963)
STRESS_SEED = 2026090993
ALPHA = 0.10

ERROR_FLOOR = np.array([
    1e-5, 1e-5, 1e-5,
    math.radians(0.01), math.radians(0.01), math.radians(0.01),
], dtype=float)
PROXY_FLOOR = np.array([
    5e-5, 5e-5, 5e-5,
    math.radians(0.05), math.radians(0.05), math.radians(0.05),
], dtype=float)
DELTA_FLOOR = np.array([
    5e-5, 5e-5, 5e-5,
    math.radians(0.02), math.radians(0.02), math.radians(0.02),
], dtype=float)
RMS_FLOOR = 1e-5


@dataclass(frozen=True)
class ShapeModel:
    beta: np.ndarray  # [6,5]
    max_iter: int


def hidden_error(ep, stage):
    """Offline training/calibration/evaluation only."""
    return np.abs(ni.pose_error(stage.R, stage.t, ep.Rg, ep.tg))


def observable_features(stage, axis, max_iter):
    """Five online-observable features for one pose coordinate."""
    return np.array([
        1.0,
        math.log(float(stage.proxy[axis]) + float(PROXY_FLOOR[axis])),
        math.log(abs(float(stage.proposed_delta[axis])) + float(DELTA_FLOOR[axis])),
        math.log(float(stage.rms) + RMS_FLOOR),
        float(stage.k) / max(1.0, float(max_iter)),
    ], dtype=float)


def fit_shape_model(episodes, max_iter):
    """Fit six log-linear error-shape regressions on TRAIN episodes only."""
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
    return ShapeModel(beta=np.asarray(betas, float), max_iter=max_iter)


def predicted_log_error(stage, axis, model):
    return float(observable_features(stage, axis, model.max_iter) @ model.beta[axis])


def episode_nonconformity(ep, model):
    """One time-uniform score per independent calibration episode.

    A_e = max_{k,i} [ log(|error|+delta_i) - m_i(observables_k) ]

    The maximum is over all retained stages and all six pose coordinates.
    """
    worst = -math.inf
    for stage in ep.stages:
        err = hidden_error(ep, stage)
        for axis in range(6):
            residual = math.log(float(err[axis]) + float(ERROR_FLOOR[axis])) - predicted_log_error(stage, axis, model)
            worst = max(worst, residual)
    return float(worst)


def split_conformal_quantile(scores, alpha=ALPHA):
    x = np.sort(np.asarray(scores, float))
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("scores must be a non-empty one-dimensional sequence")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    rank = int(math.ceil((len(x) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(x))
    return float(x[rank - 1]), rank


def calibrate(episodes, model, alpha=ALPHA):
    scores = [episode_nonconformity(ep, model) for ep in episodes]
    q, rank = split_conformal_quantile(scores, alpha)
    return {
        "alpha": float(alpha),
        "target_marginal_episode_coverage": float(1.0 - alpha),
        "episodes": len(episodes),
        "quantile_rank": rank,
        "q": q,
        "score_min": float(np.min(scores)),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
    }


def completion_bounds(stage, model, q):
    """Online completion box; uses no hidden pose error."""
    b = []
    for axis in range(6):
        upper = math.exp(predicted_log_error(stage, axis, model) + float(q)) - float(ERROR_FLOOR[axis])
        b.append(max(0.0, upper))
    return np.asarray(b, float)


def robust_task_pass(task, bounds):
    spec = ni.TASKS[task]
    b = np.asarray(bounds, float)
    if spec["kind"] == "box":
        return bool(all(float(b[i]) <= float(spec["tol"][i]) for i in spec["mask"]))
    return bool(sum(float(b[i]) / float(spec["tol"][i]) for i in spec["mask"]) <= 1.0)


def first_completion_stage(stages, task, model, q):
    """Online gate. Signature intentionally contains no Episode/Rg/tg."""
    gate_s = 0.0
    for stage in stages:
        tick = time.perf_counter()
        passed = robust_task_pass(task, completion_bounds(stage, model, q))
        gate_s += time.perf_counter() - tick
        if passed:
            return stage.k, True, gate_s
    return stages[-1].k, False, gate_s


def episode_covered(ep, model, q):
    return bool(episode_nonconformity(ep, model) <= float(q) + 1e-12)


def estimator_result(ep, task):
    k = ni.estimator_stop_stage(ep.stages)
    stage = ep.stages[k]
    ok = bool(ni.task_success(task, ni.pose_error(stage.R, stage.t, ep.Rg, ep.tg)))
    return {"k": int(k), "success": ok, "latency_s": ni.cumulative_time(ep.stages, k)}


def completion_result(ep, task, model, q):
    k, act, gate_s = first_completion_stage(ep.stages, task, model, q)
    stage = ep.stages[k]
    ok = bool(act and ni.task_success(task, ni.pose_error(stage.R, stage.t, ep.Rg, ep.tg)))
    return {
        "k": int(k), "act": bool(act), "hold": bool(not act),
        "success": ok, "unsafe_act": bool(act and not ok),
        "latency_s": ni.cumulative_time(ep.stages, k) + gate_s,
    }


def summarize_task(episodes, task, model, q):
    covered = [episode_covered(ep, model, q) for ep in episodes]
    crows = [completion_result(ep, task, model, q) for ep in episodes]
    erows = [estimator_result(ep, task) for ep in episodes]
    n = len(episodes)
    kc = np.asarray([r["k"] for r in crows], float)
    ke = np.asarray([r["k"] for r in erows], float)
    act = sum(r["act"] for r in crows)
    success = sum(r["success"] for r in crows)
    unsafe = sum(r["unsafe_act"] for r in crows)
    unsafe_while_covered = sum(bool(r["unsafe_act"] and cov) for r, cov in zip(crows, covered))
    est_success = sum(r["success"] for r in erows)
    return {
        "episodes": n,
        "completion_stop": {
            "mean_k": float(np.mean(kc)),
            "act_rate": act / n,
            "hold_rate": 1.0 - act / n,
            "completion_rate": success / n,
            "unsafe_act_rate": unsafe / n,
            "success_given_act": success / act if act else None,
            "mean_latency_ms": float(1000.0 * np.mean([r["latency_s"] for r in crows])),
        },
        "estimator_stop": {
            "mean_k": float(np.mean(ke)),
            "completion_rate": est_success / n,
            "unsafe_act_rate": 1.0 - est_success / n,
            "mean_latency_ms": float(1000.0 * np.mean([r["latency_s"] for r in erows])),
        },
        "completion_before_estimator_rate": float(np.mean(kc < ke)),
        "mean_k_completion_minus_estimator": float(np.mean(kc - ke)),
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
    lines = [
        "# Learned-shape trajectory-conformal completion results", "",
        "> **[NumericalBackend + Train/Calibrate/Test]** Generated point clouds and ICP/Kabsch only; no camera/GPU/robot.", "",
        f"Target marginal whole-trajectory coverage: {100*c['target_marginal_episode_coverage']:.1f}% (alpha={c['alpha']:.3f}); calibration episodes={c['episodes']}; q={c['q']:.6g}.", "",
        "The shape model is fitted only on TRAIN episodes. The conformal scale is computed only on disjoint CALIBRATION episodes. Final test seeds were changed after earlier development seeds were inspected.", "",
    ]
    for condition in ("in_distribution", "stress_2x_sensor_noise"):
        d = payload["evaluation"][condition]
        ci = d["trajectory_envelope_coverage_wilson95"]
        lines += [
            f"## {condition}", "",
            f"Whole-trajectory envelope coverage: **{100*d['trajectory_envelope_coverage']:.2f}%** (Wilson 95% CI {100*ci[0]:.2f}%–{100*ci[1]:.2f}%).", "",
            "| Task | estimator mean k | completion mean k | k_C<k_E | estimator completion | completion-stop completion | HOLD | unsafe ACT |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for task, s in d["tasks"].items():
            a, e = s["completion_stop"], s["estimator_stop"]
            lines.append(
                f"| {task} | {e['mean_k']:.3f} | {a['mean_k']:.3f} | {100*s['completion_before_estimator_rate']:.2f}% | "
                f"{100*e['completion_rate']:.2f}% | {100*a['completion_rate']:.2f}% | {100*a['hold_rate']:.2f}% | {100*a['unsafe_act_rate']:.2f}% |"
            )
        lines.append("")
    lines += [
        "## Mechanical implication check", "",
        "For each implemented task reader, CI verifies that an unsafe ACT never occurs on an episode whose true trajectory is contained by the calibrated envelope. This is a code-level consequence of robust PASS over the entire box plus the matching numerical outcome reader; it is not a real-robot safety certificate.", "",
        "## Evidence boundary", "",
        "Split-conformal coverage is marginal under exchangeability of calibration and future episodes. The 2x-noise stress condition is deliberately shifted and is reported empirically without transferring the in-distribution guarantee. Real RGB-D/GPU latency, physical manipulation, and reader validity remain open.", "",
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
    calibration = calibrate(cal, model, ALPHA)
    q = calibration["q"]

    final_test = []
    for seed in FINAL_TEST_SEEDS:
        final_test.extend(generate(seed, cfg["test_per_seed"], cfg["max_iter"], cfg["points"]))
    stress = generate(STRESS_SEED, cfg["stress_n"], cfg["max_iter"], cfg["points"], noise_scale=2.0)

    payload = {
        "schema_version": 1,
        "label": "[NumericalBackend+LearnedShape+SplitConformal]",
        "method": "train_shape_then_trajectory_level_split_conformal_completion_envelope",
        "evidence_boundary": {
            "generated_point_clouds": True,
            "iterative_rigid_registration_executed": True,
            "hidden_ground_truth_used_online_by_gate": False,
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
        "evaluation": {
            "in_distribution": evaluate(final_test, model, q),
            "stress_2x_sensor_noise": evaluate(stress, model, q),
        },
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "learned_conformal_completion.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_md(payload, out / "LEARNED_CONFORMAL_RESULTS.md")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("ci", "quick", "standard"), default="standard")
    ap.add_argument("--out-dir", default="results/learned_conformal")
    args = ap.parse_args()
    payload = run(args.profile, args.out_dir)
    print(json.dumps({"calibration": payload["calibration"], "evaluation": payload["evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
