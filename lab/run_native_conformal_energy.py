#!/usr/bin/env python3
"""Evaluator for the seventh real-data cycle's conformal-scaled, Keystone-
energy, decay-accumulated task admissibility rule (PROP-NATIVE-04,
lab/native_conformal_energy.py), run on the same real BOP-LMO
native-sensitivity dataset runs 5-6 used (lab/data/bop_lmo_episodes_native/).

Standalone runner, mirroring lab/run_native_persistence.py's own structure
(reuses estimator_index, cumulative_ms, boot_mean_diff, wilson from
lab/run_real_system.py, and paired_binary_noninferiority/validate_task_spec
from cqts.safety, unchanged). Runs 1-6's own evaluators and results are
untouched and independently reproducible; this is a NEW file for a NEW
decision mechanism.

Ground truth is read ONLY in `evaluate()`, after the online decision for
every checkpoint/episode has already been made -- the same offline-oracle
role T* plays for runs 1-6.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqts.safety import paired_binary_noninferiority, validate_task_spec
from lab.multicheckpoint import calibrate_checkpoints, validate_checkpoints
from lab.native_conformal_energy import ConformalEnergyConfig, first_act_checkpoint
from lab.native_sensitivity import NativeConfig
from lab.run_real_system import (
    boot_mean_diff,
    cumulative_ms,
    ensure_disjoint,
    estimator_index,
    fit_model,
    load_jsonl,
    task_pass,
    validate_dataset,
    wilson,
)


def _load_and_validate_splits(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train, cal, test = load_jsonl(args.train), load_jsonl(args.calibration), load_jsonl(args.test)
    ensure_disjoint(("train", train), ("calibration", cal), ("test", test))
    validate_dataset(train)
    validate_dataset(cal)
    validate_dataset(test)
    return cfg, train, cal, test


def evaluate(test, tasks, model, q_by_checkpoint, native_cfg, cfg_by_task, inference, repeats):
    result = {"episodes": len(test), "tasks": {}}
    n = len(test)
    for ti, (name, spec) in enumerate(tasks.items()):
        validate_task_spec(spec)
        cfg = cfg_by_task[name]
        act_k, endpoint_k, est_k = [], [], []
        act_succ, e_succ, hold, unsafe = [], [], [], []
        gate_a_only, gate_b_only, both_gates = 0, 0, 0
        degenerate_stage_count, total_stage_count = 0, 0
        for ep in test:
            idx, acted, audit = first_act_checkpoint(ep, spec, model, q_by_checkpoint, native_cfg, cfg)
            total_stage_count += len(audit)
            degenerate_stage_count += sum(1 for a in audit if a["degenerate"])
            gate_a_only += sum(1 for a in audit if a["m_k"] >= cfg.theta and not a["gate_b_certificate"])
            gate_b_only += sum(1 for a in audit if a["gate_b_certificate"] and a["m_k"] < cfg.theta)
            both_gates += sum(1 for a in audit if a["gate_b_certificate"] and a["m_k"] >= cfg.theta)
            ei = estimator_index(ep["stages"])
            end_i = idx if acted else len(ep["stages"]) - 1
            aerr = np.asarray(ep["oracle"]["abs_pose_error_6d"][end_i], float)
            eerr = np.asarray(ep["oracle"]["abs_pose_error_6d"][ei], float)
            asucc = bool(acted and task_pass(spec, aerr))
            esucc = bool(task_pass(spec, eerr))
            un = bool(acted and not asucc)
            act_k.append(ep["stages"][idx]["k"] if acted else None)
            endpoint_k.append(ep["stages"][end_i]["k"])
            est_k.append(ep["stages"][ei]["k"])
            act_succ.append(int(asucc))
            e_succ.append(int(esucc))
            hold.append(int(not acted))
            unsafe.append(int(un))

        prefix_a, prefix_e = [], []
        for ep in test:
            idx, acted, _audit = first_act_checkpoint(ep, spec, model, q_by_checkpoint, native_cfg, cfg)
            ei = estimator_index(ep["stages"])
            end_i = idx if acted else len(ep["stages"]) - 1
            prefix_a.append(cumulative_ms(ep["stages"], end_i))
            prefix_e.append(cumulative_ms(ep["stages"], ei))

        endpoint_diff = boot_mean_diff(endpoint_k, est_k, 1000 + ti, repeats)
        prefix_diff = boot_mean_diff(prefix_a, prefix_e, 3000 + ti, repeats)
        ni = paired_binary_noninferiority(
            act_succ, e_succ,
            margin=float(inference["margin"]),
            confidence_level=float(inference["confidence_level"]),
            minimum_n=int(inference["minimum_test_episodes"]),
        )
        acted_pairs = [(a, e) for a, e in zip(act_k, est_k) if a is not None]
        before = sum(a < e for a, e in acted_pairs)
        result["tasks"][name] = {
            "act_hitting_time": {
                "act_rate": float(np.mean([a is not None for a in act_k])),
                "mean_k_given_act": float(np.mean([a for a in act_k if a is not None])) if acted_pairs else None,
                "act_before_estimator_rate_all_episodes": before / n,
                "act_before_estimator_rate_given_act": before / len(acted_pairs) if acted_pairs else None,
                "act_stage_distribution": {str(k): act_k.count(k) for k in sorted(set(a for a in act_k if a is not None))},
            },
            "executed_native_conformal_energy_policy": {
                "mean_endpoint_k": float(np.mean(endpoint_k)),
                "completion_rate": float(np.mean(act_succ)),
                "hold_rate": float(np.mean(hold)),
                "unsafe_act_rate": float(np.mean(unsafe)),
                "mean_trajectory_prefix_ms": float(np.mean(prefix_a)),
            },
            "estimator_stop": {
                "mean_k": float(np.mean(est_k)),
                "completion_rate": float(np.mean(e_succ)),
                "mean_trajectory_prefix_ms": float(np.mean(prefix_e)),
            },
            "gate_diagnostics": {
                "checkpoint_readings_total": total_stage_count,
                "m_k_ge_theta_but_gate_b_false_count": gate_a_only,
                "gate_b_true_but_m_k_lt_theta_count": gate_b_only,
                "both_gates_true_count": both_gates,
            },
            "paired_endpoint_k_difference": endpoint_diff,
            "paired_prefix_perception_ms_difference": prefix_diff,
            "noninferiority": ni,
            "unsafe_act_episode_count": int(sum(unsafe)),
            "degenerate_h_k_stage_rate": degenerate_stage_count / total_stage_count if total_stage_count else None,
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="lab_results.json")
    args = ap.parse_args()

    cfg, train, cal, test = _load_and_validate_splits(args)
    ns_cfg = cfg["native_sensitivity"]
    native_cfg = NativeConfig(lambda_ref=float(ns_cfg["lambda_ref"]), cap=float(ns_cfg["cap"]))

    ce_cfg = cfg["native_conformal_energy"]
    checkpoints = tuple(int(k) for k in ce_cfg["checkpoints"])
    alpha = float(ce_cfg["alpha"])
    ceiling = float(ce_cfg["ceiling"])
    rho = float(ce_cfg["rho"])
    theta_per_task = ce_cfg["theta_per_task"]

    validate_checkpoints(list(checkpoints), train[0]["stages"])
    error_floor = cfg["error_floor"]
    model = fit_model(train, error_floor)
    q_by_checkpoint = calibrate_checkpoints(cal, model, list(checkpoints), alpha)

    cfg_by_task = {
        name: ConformalEnergyConfig(checkpoints=checkpoints, ceiling=ceiling, rho=rho, theta=float(theta_per_task[name]))
        for name in cfg["tasks"]
    }

    inference_cfg = cfg.get("inference", {})
    inference = {
        "margin": float(inference_cfg.get("noninferiority_margin", 0.05)),
        "confidence_level": float(inference_cfg.get("confidence_level", 0.95)),
        "minimum_test_episodes": int(inference_cfg.get("minimum_test_episodes", 30)),
    }
    ev = evaluate(test, cfg["tasks"], model, q_by_checkpoint, native_cfg, cfg_by_task, inference,
                  int(cfg.get("bootstrap_repeats", 5000)))

    q_report = {int(k): (None if q == float("inf") else q, rank) for k, (q, rank) in q_by_checkpoint.items()}

    payload = {
        "schema_version": 2,
        "claim": (
            "PROP-NATIVE-04: conformal-quantile-scaled perturbation magnitude + Keystone "
            "quadratic-form energy + decayed real-valued accumulator, gated by PROP-CONF-03's "
            "own per-checkpoint certificate condition -- NO ground truth is used in the online "
            "decision anywhere."
        ),
        "units": {"translation": "metres", "rotation": "radians", "time": "milliseconds"},
        "protocol": {
            "mode": "native_conformal_energy",
            "checkpoints": list(checkpoints),
            "alpha": alpha,
            "ceiling_weld_M40v1": ceiling,
            "rho": rho,
            "theta_per_task": theta_per_task,
            "q_by_checkpoint": q_report,
            "train_episodes": len(train),
            "calibration_episodes": len(cal),
            "test_episodes": len(test),
            "inference": inference,
            "reference": "PROP-NATIVE-04, ~/ANSE.ASIA/toledo/registry/proposals/native_retained_sensitivity.json",
        },
        "evaluation": ev,
        "evidence_boundary": {
            "online_gate_uses_oracle": False,
            "physical_robot_result_inferred_from_pose_error": False,
            "trajectory_prefix_timing_is_not_online_speedup": True,
            "result_scope": "instrumented iterative 6D pose backend; task readers are declared pose-error admissibility tests; oracle T* used only offline, after every online decision, to score ACT/HOLD episodes; q_k calibrated offline on CALIBRATION only, never against a test episode",
            "continuum_representation_caveat": "T_hat_k is still represented in SE(3), a continuum manifold (theory/CONTINUUM_AUDIT_20260909.md item 1); unchanged from run5/6",
            "checkpoint_indexed_accumulator_caveat": "m_k accumulates once per predeclared checkpoint (q_k is only calibrated at those 4 stages), not once per ICP iteration -- a disclosed interpretation of the registered per-stage index k (see lab/native_conformal_energy.py module docstring)",
            "certificate_condition_reading_caveat": "the registered ACT condition 'q_{k_m} <= tau_i' is read here as PROP-CONF-03's own certificate condition (completion_bounds within task tolerance), not a literal scalar comparison of a dimensionless log-nonconformity quantile to a physical tolerance (a units mismatch as typeset) -- see module docstring",
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
