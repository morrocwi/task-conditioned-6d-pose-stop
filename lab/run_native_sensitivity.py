#!/usr/bin/env python3
"""Evaluator for the fifth real-data cycle's native retained-sensitivity model
(PROP-NATIVE-01/02, lab/native_sensitivity.py).

This is a standalone runner, deliberately NOT a new `--mode` inside
`lab/run_real_system.py`: the decision mechanism has no fitted error-shape
model, no calibrated completion set, and no conformal quantile, so almost
nothing of that file's `whole_trajectory`/`decay_predictor`/
`bonferroni_multicheckpoint` machinery (fit_model, nonconformity,
conformal_quantile, completion_bounds) applies. Only a handful of
backend-agnostic utility functions are reused directly from
`lab/run_real_system.py` (estimator_index, cumulative_ms, boot_mean_diff,
wilson) and from `cqts.safety` (paired_binary_noninferiority,
validate_task_spec) -- runs 1-4's own files and code paths are completely
untouched by this script.

Online decision: `lab/native_sensitivity.py:first_act_stage`, ground-truth
free. Ground truth (`oracle.abs_pose_error_6d`) is read ONLY below, after the
decision for every stage/episode has already been made, exactly the same
offline-oracle role T* already plays for runs 1-4's own evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqts.safety import paired_binary_noninferiority, validate_task_spec
from lab.native_sensitivity import NativeConfig, first_act_stage
from lab.run_real_system import (
    boot_mean_diff,
    cumulative_ms,
    ensure_disjoint,
    estimator_index,
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


def evaluate(test, tasks, cfg: NativeConfig, inference, repeats):
    result = {"episodes": len(test), "tasks": {}}
    n = len(test)
    for ti, (name, spec) in enumerate(tasks.items()):
        validate_task_spec(spec)
        act_k, endpoint_k, est_k = [], [], []
        act_succ, e_succ, hold, unsafe = [], [], [], []
        prefix_a, prefix_e = [], []
        degenerate_stage_count, total_stage_count = 0, 0
        for ep in test:
            idx, acted, audit = first_act_stage(ep["stages"], spec, cfg)
            total_stage_count += len(audit)
            degenerate_stage_count += sum(1 for a in audit if a["degenerate"])
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
            },
            "executed_native_policy": {
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
    native_cfg = NativeConfig(lambda_ref=float(cfg["native_sensitivity"]["lambda_ref"]),
                               cap=float(cfg["native_sensitivity"]["cap"]))
    inference_cfg = cfg.get("inference", {})
    inference = {
        "margin": float(inference_cfg.get("noninferiority_margin", 0.05)),
        "confidence_level": float(inference_cfg.get("confidence_level", 0.95)),
        "minimum_test_episodes": int(inference_cfg.get("minimum_test_episodes", 30)),
    }
    ev = evaluate(test, cfg["tasks"], native_cfg, inference, int(cfg.get("bootstrap_repeats", 5000)))

    payload = {
        "schema_version": 2,
        "claim": "native retained-sensitivity ACT/CONTINUE/HOLD stopping of iterative 6D pose refinement (PROP-NATIVE-01/02, H3) -- NO ground truth is used in the online decision anywhere; no calibrated completion set, no conformal quantile",
        "units": {"translation": "metres", "rotation": "radians", "time": "milliseconds"},
        "protocol": {
            "mode": "native_sensitivity",
            "decision_rule": (
                "at stage k, select H_k's single smallest eigenvalue/eigenvector "
                "(lambda_min, v_min); perturb T_hat_k by +-m*v_min with "
                "m=min(C/lambda_min, CAP); ACT iff cqts.safety.fail_closed_task_pass "
                "gives the SAME verdict for the unperturbed pose (trivial PASS, "
                "zero self-deviation) and both perturbed candidates; else CONTINUE "
                "if stages remain, else HOLD"
            ),
            "lambda_ref_train_derived": native_cfg.lambda_ref,
            "cap": native_cfg.cap,
            "c_scale": native_cfg.c_scale,
            "train_episodes": len(train),
            "calibration_episodes": len(cal),
            "test_episodes": len(test),
            "inference": inference,
            "reference": "PROP-NATIVE-01/02, ~/ANSE.ASIA/toledo/registry/proposals/native_retained_sensitivity.json",
        },
        "evaluation": ev,
        "evidence_boundary": {
            "online_gate_uses_oracle": False,
            "physical_robot_result_inferred_from_pose_error": False,
            "trajectory_prefix_timing_is_not_online_speedup": True,
            "result_scope": "instrumented iterative 6D pose backend; task readers are declared pose-error admissibility tests; oracle T* used only offline, after every online decision, to score ACT/HOLD episodes",
            "continuum_representation_caveat": "T_hat_k is still represented in SE(3), a continuum manifold (theory/CONTINUUM_AUDIT_20260909.md item 1); this cycle removes T* from the online decision but does not fix that deeper injection",
            "mixed_units_caveat": "a single H_k eigenvector mixes 3 rotation (radians) and 3 translation (metres) components, perturbed by one shared scalar magnitude without separate nondimensionalization -- same honesty gap already disclosed for kappa in docs/NAVIER_STOKES_THROUGH_OUR_LENS.md",
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
