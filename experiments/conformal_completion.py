#!/usr/bin/env python3
"""Trajectory-calibrated completion envelopes for task-conditioned pose stopping.

This experiment turns the Toledo-derived completion-envelope idea into an
empirically calibrated numerical gate.

Evidence boundary
-----------------
EXECUTED: generated 3-D point clouds, iterative ICP/Kabsch trajectories,
          split calibration, held-out numerical evaluation.
NOT EXECUTED: camera/RGB-D sensor, neural pose model, contact physics,
              physical robot.

The gate receives only the current observable ICP proxy, the frozen calibration
scale, and the declared task. Hidden ground-truth pose error is used only to
construct the calibration score offline and to evaluate held-out outcomes.

The calibration score is episode-level: it takes the maximum normalized pose
error over every refinement stage and every pose coordinate in an episode. This
makes the resulting box an envelope for the whole retained trajectory rather
than a pointwise interval chosen after the stopping time.
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

from experiments import numerical_icp as ni

CAL_SEED = 2026090921
TEST_SEEDS = (2026090931, 2026090932, 2026090933)
STRESS_SEED = 2026090991
ALPHA = 0.10

# Fixed numerical regularizers. They are design choices, not learned truths.
FLOOR = np.array([
    0.00025, 0.00025, 0.00025,
    math.radians(0.25), math.radians(0.25), math.radians(0.25),
], dtype=float)


def stage_true_error(ep, stage):
    """Evaluation/calibration only; never an input to the online gate."""
    return np.abs(ni.pose_error(stage.R, stage.t, ep.Rg, ep.tg))


def episode_nonconformity(ep, floor=FLOOR):
    """One scalar score per independent episode.

    A_e = max_{k,i} |error_{e,k,i}| / (proxy_{e,k,i} + floor_i)

    Taking the maximum across the complete trajectory means that if a future
    episode's score is <= q, then every retained stage/axis is inside the same
    q-scaled envelope. This avoids silently treating adaptively selected stages
    as independent calibration samples.
    """
    worst = 0.0
    for s in ep.stages:
        ratio = stage_true_error(ep, s) / (np.asarray(s.proxy, float) + floor)
        worst = max(worst, float(np.max(ratio)))
    return worst


def split_conformal_quantile(scores, alpha=ALPHA):
    """Finite-sample split-conformal order statistic for exchangeable episodes."""
    x = np.sort(np.asarray(scores, float))
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("scores must be a non-empty one-dimensional sequence")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    rank = int(math.ceil((len(x) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(x))
    return float(x[rank - 1]), rank


def calibrate_envelope(episodes, alpha=ALPHA):
    scores = [episode_nonconformity(ep) for ep in episodes]
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


def completion_bounds(stage, q, floor=FLOOR):
    """Axis-aligned retained completion envelope around the current estimate."""
    return float(q) * (np.asarray(stage.proxy, float) + floor)


def robust_task_pass(task, bounds):
    """Exact robust PASS for the monotone absolute-value toy task readers."""
    spec = ni.TASKS[task]
    b = np.asarray(bounds, float)
    if spec["kind"] == "box":
        return bool(all(float(b[i]) <= float(spec["tol"][i]) for i in spec["mask"]))
    return bool(sum(float(b[i]) / float(spec["tol"][i]) for i in spec["mask"]) <= 1.0)


def first_completion_stage(ep, task, q):
    gate_s = 0.0
    for s in ep.stages:
        tick = time.perf_counter()
        ok = robust_task_pass(task, completion_bounds(s, q))
        gate_s += time.perf_counter() - tick
        if ok:
            return s.k, True, gate_s
    return ep.stages[-1].k, False, gate_s


def episode_covered(ep, q):
    """Whether the hidden trajectory lies inside the frozen completion envelope."""
    for s in ep.stages:
        if np.any(stage_true_error(ep, s) > completion_bounds(s, q) + 1e-15):
            return False
    return True


def eval_task_stop(ep, task, q):
    k, act, gate_s = first_completion_stage(ep, task, q)
    st = ep.stages[k]
    err = ni.pose_error(st.R, st.t, ep.Rg, ep.tg)
    success = bool(act and ni.task_success(task, err))
    return {
        "k": int(k),
        "act": bool(act),
        "hold": bool(not act),
        "success": success,
        "unsafe_act": bool(act and not success),
        "latency_s": ni.cumulative_time(ep.stages, k) + gate_s,
    }


def eval_estimator_stop(ep, task):
    k = ni.estimator_stop_stage(ep.stages)
    st = ep.stages[k]
    err = ni.pose_error(st.R, st.t, ep.Rg, ep.tg)
    success = bool(ni.task_success(task, err))
    return {
        "k": int(k), "act": True, "hold": False, "success": success,
        "unsafe_act": bool(not success),
        "latency_s": ni.cumulative_time(ep.stages, k),
    }


def summarize_task(episodes, task, q):
    task_rows = [eval_task_stop(ep, task, q) for ep in episodes]
    est_rows = [eval_estimator_stop(ep, task) for ep in episodes]
    n = len(episodes)
    kt = np.array([r["k"] for r in task_rows], float)
    ke = np.array([r["k"] for r in est_rows], float)
    act = sum(r["act"] for r in task_rows)
    succ = sum(r["success"] for r in task_rows)
    unsafe = sum(r["unsafe_act"] for r in task_rows)
    est_succ = sum(r["success"] for r in est_rows)
    return {
        "episodes": n,
        "task_stop": {
            "mean_k": float(np.mean(kt)),
            "act_rate": act / n,
            "hold_rate": 1.0 - act / n,
            "completion_rate": succ / n,
            "unsafe_act_rate": unsafe / n,
            "success_given_act": succ / act if act else None,
            "mean_latency_ms": float(1000.0 * np.mean([r["latency_s"] for r in task_rows])),
        },
        "estimator_stop": {
            "mean_k": float(np.mean(ke)),
            "completion_rate": est_succ / n,
            "unsafe_act_rate": 1.0 - est_succ / n,
            "mean_latency_ms": float(1000.0 * np.mean([r["latency_s"] for r in est_rows])),
        },
        "task_before_estimator_rate": float(np.mean(kt < ke)),
        "mean_k_task_minus_estimator": float(np.mean(kt - ke)),
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
    lines = [
        "# Trajectory-calibrated completion-envelope experiment", "",
        "> **[NumericalBackend + SplitCalibration]** This is a generated-point-cloud ICP/Kabsch experiment, not a camera/GPU/robot experiment.", "",
        f"Frozen episode-level calibration: alpha={c['alpha']:.3f}, target marginal episode coverage={100*c['target_marginal_episode_coverage']:.1f}%, n={c['episodes']}, q={c['q']:.6g}.", "",
        "The nonconformity score is the maximum normalized hidden pose error across **all stages and all six coordinates in one episode**. Calibration therefore treats the episode, not the adaptively selected refinement stage, as the exchangeable unit.", "",
    ]
    for condition in ("in_distribution", "stress_2x_sensor_noise"):
        d = payload["evaluation"][condition]
        ci = d["trajectory_envelope_coverage_wilson95"]
        lines += [
            f"## {condition}", "",
            f"Trajectory-envelope coverage: **{100*d['trajectory_envelope_coverage']:.2f}%** (Wilson 95% CI {100*ci[0]:.2f}%–{100*ci[1]:.2f}%).", "",
            "| Task | estimator mean k | task mean k | k_task<k_est | estimator completion | task completion | HOLD | unsafe ACT |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for task, s in d["tasks"].items():
            t, e = s["task_stop"], s["estimator_stop"]
            lines.append(
                f"| {task} | {e['mean_k']:.3f} | {t['mean_k']:.3f} | {100*s['task_before_estimator_rate']:.2f}% | "
                f"{100*e['completion_rate']:.2f}% | {100*t['completion_rate']:.2f}% | {100*t['hold_rate']:.2f}% | {100*t['unsafe_act_rate']:.2f}% |"
            )
        lines.append("")
    lines += [
        "## Evidence boundary", "",
        "The split-conformal interpretation requires exchangeability of calibration and future episodes. The public stress condition deliberately violates the in-distribution generator by doubling sensor noise; coverage and stopping performance under that shift are empirical readouts, not covered by the in-distribution guarantee.", "",
        "The completion box is a retained computational readout, not the hidden world. Real RGB-D/GPU acceleration, physical manipulation non-inferiority, and safety certification remain HOLD/open hypotheses.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(profile, out_dir):
    if profile == "ci":
        cfg = dict(cal_n=28, test_per_seed=6, stress_n=12, max_iter=12, points=56)
    elif profile == "quick":
        cfg = dict(cal_n=64, test_per_seed=14, stress_n=24, max_iter=16, points=80)
    else:
        cfg = dict(cal_n=160, test_per_seed=40, stress_n=60, max_iter=20, points=120)

    cal = generate(CAL_SEED, cfg["cal_n"], cfg["max_iter"], cfg["points"])
    calibration = calibrate_envelope(cal, ALPHA)
    q = calibration["q"]

    test = []
    for seed in TEST_SEEDS:
        test.extend(generate(seed, cfg["test_per_seed"], cfg["max_iter"], cfg["points"]))
    stress = generate(STRESS_SEED, cfg["stress_n"], cfg["max_iter"], cfg["points"], noise_scale=2.0)

    payload = {
        "schema_version": 1,
        "label": "[NumericalBackend+SplitCalibration]",
        "method": "trajectory_level_split_conformal_completion_envelope",
        "evidence_boundary": {
            "generated_point_clouds": True,
            "iterative_rigid_registration_executed": True,
            "hidden_ground_truth_used_online_by_gate": False,
            "camera_executed": False,
            "rgbd_sensor_executed": False,
            "neural_pose_model_executed": False,
            "physical_robot_executed": False,
        },
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "configuration": {**cfg, "alpha": ALPHA, "floor": FLOOR.tolist(), "cal_seed": CAL_SEED, "test_seeds": list(TEST_SEEDS), "stress_seed": STRESS_SEED},
        "calibration": calibration,
        "evaluation": {
            "in_distribution": evaluate(test, q),
            "stress_2x_sensor_noise": evaluate(stress, q),
        },
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "conformal_completion.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_md(payload, out / "CONFORMAL_COMPLETION_RESULTS.md")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("ci", "quick", "standard"), default="standard")
    ap.add_argument("--out-dir", default="results/conformal")
    args = ap.parse_args()
    payload = run(args.profile, args.out_dir)
    concise = {"calibration": payload["calibration"], "evaluation": payload["evaluation"]}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
