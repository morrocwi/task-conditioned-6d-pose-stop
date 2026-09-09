#!/usr/bin/env python3
"""Repeated independent numerical replication cycles for the frozen learned method.

This script does NOT tune the learned completion method. Each replicate executes
an entire TRAIN -> CALIBRATION -> TEST cycle with disjoint deterministic seeds,
then records realized whole-trajectory coverage and task-stopping utility.

Purpose: distinguish one realized final-test coverage proportion from the
behavior of the frozen procedure across repeated calibration/test draws from the
same declared numerical generator.
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
    return {
        "train": base + 11,
        "calibration": base + 21,
        "test": base + 31,
    }


def one_replicate(rep, train_n, cal_n, test_n, max_iter, points):
    seeds = seeds_for(rep)
    train = lc.generate(seeds["train"], train_n, max_iter, points)
    model = lc.fit_shape_model(train, max_iter)
    cal = lc.generate(seeds["calibration"], cal_n, max_iter, points)
    calibration = lc.calibrate(cal, model, lc.ALPHA)
    test = lc.generate(seeds["test"], test_n, max_iter, points)
    ev = lc.evaluate(test, model, calibration["q"])
    return {
        "replicate": rep,
        "seeds": seeds,
        "calibration_q": calibration["q"],
        "trajectory_coverage": ev["trajectory_envelope_coverage"],
        "tasks": {
            task: {
                "completion_before_estimator_rate": d["completion_before_estimator_rate"],
                "completion_difference": d["completion_difference_completion_minus_estimator"],
                "hold_rate": d["completion_stop"]["hold_rate"],
                "unsafe_act_rate": d["completion_stop"]["unsafe_act_rate"],
                "mean_k_completion_minus_estimator": d["mean_k_completion_minus_estimator"],
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
            "mean_completion_before_estimator_rate": statistics.mean(x["completion_before_estimator_rate"] for x in rr),
            "mean_completion_difference": statistics.mean(x["completion_difference"] for x in rr),
            "mean_hold_rate": statistics.mean(x["hold_rate"] for x in rr),
            "mean_unsafe_act_rate": statistics.mean(x["unsafe_act_rate"] for x in rr),
            "mean_k_completion_minus_estimator": statistics.mean(x["mean_k_completion_minus_estimator"] for x in rr),
            "total_unsafe_act_on_covered_episode_count": sum(x["unsafe_act_on_covered_episode_count"] for x in rr),
        }
    return out


def render_md(payload, path):
    s = payload["summary"]
    lines = [
        "# Repeated TRAIN–CALIBRATION–TEST numerical replication", "",
        "> Frozen learned-shape method; generated ICP/Kabsch backend only.", "",
        f"Replicates: **{s['replicates']}**; nominal marginal whole-trajectory coverage: **{100*s['nominal_coverage']:.1f}%**.", "",
        f"Mean realized trajectory coverage: **{100*s['mean_realized_trajectory_coverage']:.2f}%**; median **{100*s['median_realized_trajectory_coverage']:.2f}%**; range **{100*s['min_realized_trajectory_coverage']:.2f}%–{100*s['max_realized_trajectory_coverage']:.2f}%**.", "",
        f"Replicates at or above the nominal proportion: **{s['replicates_at_or_above_nominal']}/{s['replicates']}**.", "",
        "A single realized test proportion is not the conformal theorem. This repetition is an empirical diagnostic of the frozen numerical procedure across independent draws from the same declared generator.", "",
        "| Task | mean k_C<k_E | mean k_C-k_E | mean completion difference | mean HOLD | mean unsafe ACT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, d in s["tasks"].items():
        lines.append(
            f"| {task} | {100*d['mean_completion_before_estimator_rate']:.2f}% | {d['mean_k_completion_minus_estimator']:.3f} | "
            f"{100*d['mean_completion_difference']:.2f} pp | {100*d['mean_hold_rate']:.2f}% | {100*d['mean_unsafe_act_rate']:.2f}% |"
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

    rows = [one_replicate(rep, cfg["train_n"], cfg["cal_n"], cfg["test_n"], cfg["max_iter"], cfg["points"])
            for rep in range(cfg["replicates"])]
    summary = summarize(rows)
    if any(v["total_unsafe_act_on_covered_episode_count"] != 0 for v in summary["tasks"].values()):
        raise AssertionError("coverage-to-task implication violated in replication study")

    payload = {
        "schema_version": 1,
        "label": "[NumericalReplication]",
        "method": "frozen_learned_shape_repeated_train_calibrate_test",
        "configuration": cfg,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "seed_rule": "BASE_SEED + 100*rep + {11 train,21 calibration,31 test}",
        "rows": rows,
        "summary": summary,
        "evidence_boundary": {
            "same_declared_numerical_generator": True,
            "method_tuned_inside_replication": False,
            "camera_executed": False,
            "neural_pose_model_executed": False,
            "physical_robot_executed": False
        }
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "replication_study.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_md(payload, out / "REPLICATION_RESULTS.md")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=("ci", "quick", "standard"), default="standard")
    ap.add_argument("--out-dir", default="results/replication")
    args = ap.parse_args()
    payload = run(args.profile, args.out_dir)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
