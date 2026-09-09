#!/usr/bin/env python3
"""Repeated reduced-configuration numerical cycles for the frozen learned method.

This study is explicitly NOT a repetition of the first 160/160/120, K=20,
120-point final-test configuration. It is a reduced computational stress-test
of the same frozen procedure across repeated TRAIN->CALIBRATION->TEST draws.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import learned_conformal_completion as lc

BASE_SEED = 2026092000


def seeds_for(rep):
    base = BASE_SEED + 100 * rep
    return {"train": base + 11, "calibration": base + 21, "test": base + 31}


def one_replicate(rep, train_n, cal_n, test_n, max_iter, points):
    seeds = seeds_for(rep)
    train = lc.generate(seeds["train"], train_n, max_iter, points)
    model = lc.fit_shape_model(train, max_iter)
    cal = lc.generate(seeds["calibration"], cal_n, max_iter, points)
    calibration, q = lc.calibrate(cal, model, lc.ALPHA)
    test = lc.generate(seeds["test"], test_n, max_iter, points)
    ev = lc.evaluate(test, model, q)
    return {
        "replicate": rep,
        "seeds": seeds,
        "calibration_q": None if calibration["calibration_unbounded"] else calibration["q"],
        "calibration_unbounded": calibration["calibration_unbounded"],
        "trajectory_coverage": ev["trajectory_envelope_coverage"],
        "tasks": {
            task: {
                "certificate_before_estimator_rate_all_episodes": d["certificate"]["before_estimator_rate_all_episodes"],
                "certificate_rate": d["certificate"]["rate"],
                "completion_difference": d["completion_difference_completion_minus_estimator"],
                "hold_rate": d["completion_stop"]["hold_rate"],
                "unsafe_act_rate": d["completion_stop"]["unsafe_act_rate"],
                "mean_endpoint_k_completion_minus_estimator": d["mean_endpoint_k_completion_minus_estimator"],
                "unsafe_act_on_covered_episode_count": d["unsafe_act_on_covered_episode_count"],
            }
            for task, d in ev["tasks"].items()
        },
    }


def summarize(rows):
    cov = [r["trajectory_coverage"] for r in rows]
    out = {
        "replicates": len(rows),
        "nominal_coverage": 1.0 - lc.ALPHA,
        "mean_realized_trajectory_coverage": statistics.mean(cov),
        "median_realized_trajectory_coverage": statistics.median(cov),
        "min_realized_trajectory_coverage": min(cov),
        "max_realized_trajectory_coverage": max(cov),
        "replicates_at_or_above_nominal": sum(x >= 1.0 - lc.ALPHA for x in cov),
        "tasks": {},
    }
    for task in lc.ni.TASKS:
        rr = [r["tasks"][task] for r in rows]
        out["tasks"][task] = {
            "mean_certificate_before_estimator_rate_all_episodes": statistics.mean(x["certificate_before_estimator_rate_all_episodes"] for x in rr),
            "mean_certificate_rate": statistics.mean(x["certificate_rate"] for x in rr),
            "mean_completion_difference": statistics.mean(x["completion_difference"] for x in rr),
            "mean_hold_rate": statistics.mean(x["hold_rate"] for x in rr),
            "mean_unsafe_act_rate": statistics.mean(x["unsafe_act_rate"] for x in rr),
            "mean_endpoint_k_completion_minus_estimator": statistics.mean(x["mean_endpoint_k_completion_minus_estimator"] for x in rr),
            "total_unsafe_act_on_covered_episode_count": sum(x["unsafe_act_on_covered_episode_count"] for x in rr),
        }
    return out


def render_md(payload, path):
    s, cfg = payload["summary"], payload["configuration"]
    lines = [
        "# Reduced-configuration repeated TRAIN–CALIBRATION–TEST diagnostic", "",
        "> Generated ICP/Kabsch backend only. This is not a repetition of the first full standard final-test configuration.", "",
        f"Configuration: train={cfg['train_n']}, calibration={cfg['cal_n']}, test={cfg['test_n']}, max_iter={cfg['max_iter']}, points={cfg['points']}, replicates={cfg['replicates']}.", "",
        f"Mean realized trajectory coverage: **{100*s['mean_realized_trajectory_coverage']:.2f}%**; median **{100*s['median_realized_trajectory_coverage']:.2f}%**; range **{100*s['min_realized_trajectory_coverage']:.2f}%–{100*s['max_realized_trajectory_coverage']:.2f}%**.", "",
        "These cycles characterize only this reduced configuration; they must not be used to reinterpret the first 160/160/120, K=20, 120-point final-test realization.", "",
        "| Task | mean cert<est (all eps) | mean certificate rate | mean endpoint k diff | mean completion diff | mean HOLD | mean unsafe ACT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, d in s["tasks"].items():
        lines.append(
            f"| {task} | {100*d['mean_certificate_before_estimator_rate_all_episodes']:.2f}% | {100*d['mean_certificate_rate']:.2f}% | "
            f"{d['mean_endpoint_k_completion_minus_estimator']:.3f} | {100*d['mean_completion_difference']:.2f} pp | {100*d['mean_hold_rate']:.2f}% | {100*d['mean_unsafe_act_rate']:.2f}% |"
        )
    lines += ["", "For every task the workflow requires `total_unsafe_act_on_covered_episode_count == 0`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(profile, out_dir):
    if profile == "ci":
        cfg = dict(replicates=3, train_n=24, cal_n=24, test_n=18, max_iter=10, points=48)
    elif profile == "quick":
        cfg = dict(replicates=5, train_n=48, cal_n=48, test_n=36, max_iter=14, points=64)
    else:
        cfg = dict(replicates=10, train_n=80, cal_n=80, test_n=60, max_iter=16, points=80)

    rows = [one_replicate(rep, cfg["train_n"], cfg["cal_n"], cfg["test_n"], cfg["max_iter"], cfg["points"]) for rep in range(cfg["replicates"])]
    summary = summarize(rows)
    if any(v["total_unsafe_act_on_covered_episode_count"] != 0 for v in summary["tasks"].values()):
        raise AssertionError("coverage-to-task implication violated in replication study")

    payload = {
        "schema_version": 2,
        "label": "[NumericalReplication-ReducedConfiguration]",
        "method": "frozen_learned_shape_repeated_train_calibrate_test_reduced_configuration",
        "configuration": cfg,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "seed_rule": "BASE_SEED + 100*rep + {11 train,21 calibration,31 test}",
        "rows": rows,
        "summary": summary,
        "evidence_boundary": {
            "same_declared_numerical_generator_family": True,
            "same_as_first_full_standard_configuration": False,
            "method_tuned_inside_replication": False,
            "camera_executed": False,
            "neural_pose_model_executed": False,
            "physical_robot_executed": False,
        },
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "replication_study.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    render_md(payload, out / "REPLICATION_RESULTS.md")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("ci", "quick", "standard"), default="standard")
    ap.add_argument("--out-dir", default="results/replication")
    args = ap.parse_args()
    payload = run(args.profile, args.out_dir)
    print(json.dumps(payload["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
