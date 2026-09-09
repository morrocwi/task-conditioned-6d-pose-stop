#!/usr/bin/env python3
"""Same-split matched comparison for the central completion certificate.

This post-review experiment uses comparison-test seeds that are distinct from
all previously reported final tests. All methods share the same TRAIN,
CALIBRATION, TEST episodes, task readers, and numerical outcome oracle.

IMPORTANT: comparator candidate grids are selected deterministically from
CALIBRATION values only before TEST is evaluated. No test-outcome tuning occurs.

Evidence boundary: generated point clouds + numerical ICP/Kabsch only.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqts.safety import clopper_pearson_upper
from experiments import conformal_completion as raw
from experiments import learned_conformal_completion as lc
from experiments import numerical_icp as ni

COMPARISON_TEST_SEEDS = (2026091061, 2026091062, 2026091063)
RISK_TARGET = lc.ALPHA
RISK_CONFIDENCE = 0.95
MIN_ACT_RATE = 0.50
MAX_THRESHOLD_CANDIDATES = 128


def candidate_grid(values, max_candidates=MAX_THRESHOLD_CANDIDATES):
    """Deterministic monotone grid derived only from calibration values."""
    x = np.unique(np.asarray(list(values), dtype=float))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return []
    if x.size <= max_candidates:
        return [float(v) for v in x]
    idx = np.unique(np.linspace(0, x.size - 1, max_candidates, dtype=int))
    return [float(x[i]) for i in idx]


def outcome(ep, task, endpoint_k, act):
    stage = ep.stages[int(endpoint_k)]
    ok = bool(act and ni.task_success(task, ni.pose_error(stage.R, stage.t, ep.Rg, ep.tg)))
    return {
        "endpoint_k": int(endpoint_k),
        "act": bool(act),
        "hold": bool(not act),
        "success": ok,
        "unsafe_act": bool(act and not ok),
        "trajectory_prefix_ms": 1000.0 * ni.cumulative_time(ep.stages, int(endpoint_k)),
    }


def full_policy(ep, task):
    return outcome(ep, task, ep.stages[-1].k, True)


def estimator_policy(ep, task):
    return outcome(ep, task, ni.estimator_stop_stage(ep.stages), True)


def fixed_policy(ep, task, k):
    return outcome(ep, task, min(int(k), ep.stages[-1].k), True)


def central_task_score(stage, task, model):
    bounds = lc.completion_bounds(stage, model, 0.0)
    spec = ni.TASKS[task]
    if spec["kind"] == "box":
        return max(float(bounds[i]) / float(spec["tol"][i]) for i in spec["mask"])
    return sum(float(bounds[i]) / float(spec["tol"][i]) for i in spec["mask"])


def scalar_policy(ep, task, model, threshold):
    if threshold is None:
        return outcome(ep, task, ep.stages[-1].k, False)
    for stage in ep.stages:
        if central_task_score(stage, task, model) <= float(threshold):
            return outcome(ep, task, stage.k, True)
    return outcome(ep, task, ep.stages[-1].k, False)


def estimator_score(stage):
    d = np.abs(np.asarray(stage.proposed_delta, float))
    return max(float(np.max(d[:3] / 0.0005)), float(np.max(d[3:] / math.radians(0.1))))


def tuned_estimator_policy(ep, task, threshold):
    prev = False
    for stage in ep.stages[:-1]:
        small = estimator_score(stage) <= float(threshold)
        if small and prev:
            return outcome(ep, task, stage.k, True)
        prev = small
    return outcome(ep, task, ep.stages[-1].k, True)


def learned_certificate_policy(ep, task, model, q):
    k, act, _ = lc.first_completion_stage(ep.stages, task, model, q)
    return outcome(ep, task, k if act else ep.stages[-1].k, act)


def raw_certificate_policy(ep, task, q):
    k, act, _ = raw.first_completion_stage(ep.stages, task, q)
    return outcome(ep, task, k if act else ep.stages[-1].k, act)


def risk_upper(rows, confidence=RISK_CONFIDENCE):
    n = len(rows)
    unsafe = sum(r["unsafe_act"] for r in rows)
    return clopper_pearson_upper(unsafe, n, 1.0 - float(confidence))


def qualify(rows):
    return {
        "unsafe_upper": risk_upper(rows),
        "act_rate": float(np.mean([r["act"] for r in rows])),
        "mean_endpoint_k": float(np.mean([r["endpoint_k"] for r in rows])),
    }


def choose_fixed(cal, task):
    valid = []
    for k in range(cal[0].stages[-1].k + 1):
        rows = [fixed_policy(ep, task, k) for ep in cal]
        q = qualify(rows)
        if q["unsafe_upper"] <= RISK_TARGET:
            valid.append((q["mean_endpoint_k"], k, q))
    if not valid:
        return {"status": "UNQUALIFIED", "k": cal[0].stages[-1].k}
    _, k, q = min(valid)
    return {"status": "QUALIFIED", "k": int(k), "calibration": q}


def choose_scalar(cal, task, model):
    vals = (central_task_score(stage, task, model) for ep in cal for stage in ep.stages)
    candidates = candidate_grid(vals)
    valid = []
    for th in candidates:
        rows = [scalar_policy(ep, task, model, th) for ep in cal]
        q = qualify(rows)
        if q["unsafe_upper"] <= RISK_TARGET and q["act_rate"] >= MIN_ACT_RATE:
            valid.append((q["mean_endpoint_k"], -q["act_rate"], float(th), q))
    if not valid:
        return {"status": "UNQUALIFIED", "threshold": None, "candidates_checked": len(candidates)}
    _, _, th, q = min(valid)
    return {"status": "QUALIFIED", "threshold": th, "calibration": q, "candidates_checked": len(candidates)}


def choose_estimator_threshold(cal, task):
    vals = (estimator_score(stage) for ep in cal for stage in ep.stages[:-1])
    candidates = candidate_grid(vals)
    valid = []
    for th in candidates:
        rows = [tuned_estimator_policy(ep, task, th) for ep in cal]
        q = qualify(rows)
        if q["unsafe_upper"] <= RISK_TARGET:
            valid.append((q["mean_endpoint_k"], float(th), q))
    if not valid:
        return {"status": "UNQUALIFIED", "threshold": 1.0, "candidates_checked": len(candidates)}
    _, th, q = min(valid)
    return {"status": "QUALIFIED", "threshold": th, "calibration": q, "candidates_checked": len(candidates)}


def summarize(rows):
    n = len(rows)
    return {
        "episodes": n,
        "mean_endpoint_k": float(np.mean([r["endpoint_k"] for r in rows])),
        "completion_rate": float(np.mean([r["success"] for r in rows])),
        "act_rate": float(np.mean([r["act"] for r in rows])),
        "hold_rate": float(np.mean([r["hold"] for r in rows])),
        "unsafe_act_rate": float(np.mean([r["unsafe_act"] for r in rows])),
        "unsafe_act_wilson95": ni.wilson(sum(r["unsafe_act"] for r in rows), n),
        "mean_trajectory_prefix_ms": float(np.mean([r["trajectory_prefix_ms"] for r in rows])),
    }


def run(profile, out_dir):
    if profile == "ci":
        cfg = dict(train_n=36, cal_n=36, test_per_seed=6, max_iter=12, points=56)
    elif profile == "quick":
        cfg = dict(train_n=72, cal_n=72, test_per_seed=14, max_iter=16, points=80)
    else:
        cfg = dict(train_n=160, cal_n=160, test_per_seed=40, max_iter=20, points=120)

    train = lc.generate(lc.TRAIN_SEED, cfg["train_n"], cfg["max_iter"], cfg["points"])
    model = lc.fit_shape_model(train, cfg["max_iter"])
    cal = lc.generate(lc.CAL_SEED, cfg["cal_n"], cfg["max_iter"], cfg["points"])
    learned_cal, learned_q = lc.calibrate(cal, model, lc.ALPHA)
    raw_cal, raw_q = raw.calibrate_envelope(cal, raw.ALPHA)

    # Freeze all comparator parameters before opening comparison TEST.
    frozen = {}
    for task in ni.TASKS:
        frozen[task] = {
            "fixed": choose_fixed(cal, task),
            "estimator_threshold": choose_estimator_threshold(cal, task),
            "learned_scalar": choose_scalar(cal, task, model),
        }

    test = []
    for seed in COMPARISON_TEST_SEEDS:
        test.extend(lc.generate(seed, cfg["test_per_seed"], cfg["max_iter"], cfg["points"]))

    task_results = {}
    for task in ni.TASKS:
        fixed = frozen[task]["fixed"]
        scalar = frozen[task]["learned_scalar"]
        est_tuned = frozen[task]["estimator_threshold"]
        policies = {
            "full_budget": [full_policy(ep, task) for ep in test],
            "estimator_default": [estimator_policy(ep, task) for ep in test],
            "fixed_risk_checked": [fixed_policy(ep, task, fixed["k"]) for ep in test],
            "estimator_threshold_risk_checked": [tuned_estimator_policy(ep, task, est_tuned["threshold"]) for ep in test],
            "learned_scalar_risk_checked": [scalar_policy(ep, task, model, scalar["threshold"]) for ep in test],
            "raw_completion_envelope": [raw_certificate_policy(ep, task, raw_q) for ep in test],
            "trajectory_conformal_completion": [learned_certificate_policy(ep, task, model, learned_q) for ep in test],
        }
        task_results[task] = {"tuning": frozen[task], "policies": {name: summarize(rows) for name, rows in policies.items()}}

    payload = {
        "schema_version": 2,
        "label": "[MatchedComparison-NewTestSeeds]",
        "configuration": {
            **cfg,
            "risk_target": RISK_TARGET,
            "risk_confidence": RISK_CONFIDENCE,
            "min_act_rate_scalar": MIN_ACT_RATE,
            "max_threshold_candidates": MAX_THRESHOLD_CANDIDATES,
        },
        "lineage": {
            "train_seed": lc.TRAIN_SEED,
            "calibration_seed": lc.CAL_SEED,
            "comparison_test_seeds": list(COMPARISON_TEST_SEEDS),
            "previous_final_test_seeds_excluded": list(lc.FINAL_TEST_SEEDS),
            "test_opened_only_after_comparator_parameters_frozen_in_run": True,
        },
        "learned_calibration": learned_cal,
        "raw_calibration": raw_cal,
        "tasks": task_results,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "evidence_boundary": {
            "generated_point_clouds": True,
            "matched_train_calibration_test_episodes": True,
            "new_test_seeds_not_in_previous_final_test": True,
            "risk_checked_baselines_use_calibration_only": True,
            "test_outcomes_used_for_tuning": False,
            "trajectory_prefix_timing_is_online_speedup": False,
            "camera_executed": False,
            "physical_robot_executed": False,
        },
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_baselines.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("ci", "quick", "standard"), default="standard")
    ap.add_argument("--out-dir", default="results/matched_baselines")
    args = ap.parse_args()
    p = run(args.profile, args.out_dir)
    print(json.dumps({task: data["policies"] for task, data in p["tasks"].items()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
