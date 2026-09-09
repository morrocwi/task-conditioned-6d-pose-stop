"""Memory/persistence accumulator on top of the native retained-sensitivity
invariance check (PROP-NATIVE-03).

Sixth real-data cycle (ops/HANDOFF_2026-09-09_external_dataset.md, "Run 6
authorization"). Toledo proposal:
~/ANSE.ASIA/toledo/registry/proposals/native_retained_sensitivity.json,
PROP-NATIVE-03 (registered before this file was written -- Toledo-first).

Registered rule
----------------
    M_k := M_{k-1} + 1[task verdict invariant under every imagined
                        perturbation at stage k],   M_0 := 0
    ACT <=> M_k >= theta   (a predeclared persistence threshold)

Interpretation choice, disclosed (the registered LaTeX statement is literally
a running SUM that never decreases; the registered `note_ascii` in the same
Toledo entry describes it as "a streak count of consecutive stages" and the
proposal's own name is "a persistent streak of task-invariance, not a
single-instant check"). A non-resetting running sum and a resetting
consecutive-streak counter agree whenever the per-stage indicator never
turns back to 0 after having been 1 -- but they diverge the moment a single
stage's invariance check fails after a prior pass. Per the note_ascii and the
proposal name, THIS implementation uses the resetting streak interpretation:

    M_k = M_{k-1} + 1   if invariant at stage k
    M_k = 0             otherwise

This is the interpretation that actually does the job the proposal names
("persistent streak", "noise-robustness" against a single spurious pass) --
a non-resetting cumulative count would let one early pass among many later
failures still reach theta and ACT, which does not filter noise the way the
proposal's own honest_caveats describe. Stated here explicitly rather than
silently picked.

Reuses, unmodified
------------------
`lab/native_sensitivity.py:stage_decision` (PROP-NATIVE-02's single-instant
invariance check, including its `NativeConfig`, `select_least_constrained_
direction`, `perturbation_magnitude`, `perturbed_pose`, `deviation_axes`) is
called once per stage, exactly as run 5 called it. This module ONLY adds the
M_k accumulator and the theta-gated ACT rule on top -- it does not touch
run 5's own decision function, config, or data files, per the handoff's
explicit "keep run1-5 code paths intact" instruction. Run 5's own evaluator
(`lab/run_native_sensitivity.py`) and results remain independently
reproducible.

Honest structural note (see RATIONALE.md for the TRAIN diagnostic that
produced it): on TRAIN data, `stage_decision(...)["act"]` (the single-instant
check) evaluates True at EVERY stage of EVERY TRAIN episode for all three
tasks (640/640 stage readings, 40/40 initial streaks of full length 16).
Under path (a) below (PROP-NATIVE-02's perturbation rule reused byte-for-
byte), the streak accumulator therefore has nothing to threshold against --
M_k is deterministically k+1 for every episode, and ACT fires at the fixed
stage k=theta-1 regardless of episode content. This is disclosed as a
predicted structural finding BEFORE running against test.jsonl, not
discovered after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lab.native_sensitivity import NativeConfig, stage_decision


@dataclass(frozen=True)
class PersistenceConfig:
    theta: int
    residual_scale_ref: float | None = None  # None => path (a); set => path (b)
    residual_cap_multiplier: float | None = None  # only used when residual_scale_ref is set


def _residual_rmse_from_features(features) -> float:
    """features[1] = log(ICP residual RMSE + 1e-9) (lab/bop_icp_backend.py's
    observable_feature_vector, index 1, identical convention reused from
    runs 1-5's own online feature export). Returns the RMSE itself."""
    return float(np.exp(float(features[1])) - 1e-9)


def stage_decision_with_variant(R, t, H, spec: dict, native_cfg: NativeConfig,
                                 pcfg: PersistenceConfig, features=None) -> dict:
    """PROP-NATIVE-02's single-instant check, optionally with path (b)'s
    residual-scaled perturbation magnitude substituted in place of the
    unmodified rule. When pcfg.residual_scale_ref is None this is IDENTICAL
    to lab.native_sensitivity.stage_decision (path a)."""
    if pcfg.residual_scale_ref is None:
        return stage_decision(R, t, H, spec, native_cfg)

    # Path (b): scale the perturbation magnitude up when the CURRENT
    # observable ICP residual RMSE exceeds its TRAIN-typical (reference)
    # value -- disclosed, additive variant of PROP-NATIVE-02's rule, not a
    # replacement of PROP-NATIVE-03's own accumulator logic.
    from lab.native_sensitivity import (
        deviation_axes,
        perturbation_magnitude,
        perturbed_pose,
        select_least_constrained_direction,
    )
    from cqts.safety import fail_closed_task_pass

    lambda_min, v_min = select_least_constrained_direction(H)
    magnitude, degenerate = perturbation_magnitude(lambda_min, native_cfg)

    residual_rmse = _residual_rmse_from_features(features) if features is not None else pcfg.residual_scale_ref
    scale = max(1.0, residual_rmse / pcfg.residual_scale_ref) if pcfg.residual_scale_ref > 0 else 1.0
    cap_b = pcfg.residual_cap_multiplier * native_cfg.cap if pcfg.residual_cap_multiplier else native_cfg.cap
    magnitude_b = min(magnitude * scale, cap_b)

    baseline_dev = np.zeros(6, dtype=float)
    verdict_baseline = bool(fail_closed_task_pass(spec, baseline_dev))

    R_plus, t_plus = perturbed_pose(R, t, v_min, magnitude_b, +1.0)
    R_minus, t_minus = perturbed_pose(R, t, v_min, magnitude_b, -1.0)
    dev_plus = deviation_axes(R, t, R_plus, t_plus)
    dev_minus = deviation_axes(R, t, R_minus, t_minus)
    verdict_plus = bool(fail_closed_task_pass(spec, dev_plus))
    verdict_minus = bool(fail_closed_task_pass(spec, dev_minus))

    act = verdict_baseline and verdict_plus and verdict_minus
    return {
        "act": bool(act),
        "lambda_min": lambda_min,
        "magnitude": magnitude_b,
        "magnitude_base": magnitude,
        "residual_scale_applied": scale,
        "degenerate": bool(degenerate),
        "verdict_baseline": verdict_baseline,
        "verdict_plus": verdict_plus,
        "verdict_minus": verdict_minus,
        "deviation_plus": [float(x) for x in dev_plus],
        "deviation_minus": [float(x) for x in dev_minus],
    }


def persistent_first_act_stage(stages: list[dict], spec: dict, native_cfg: NativeConfig,
                                pcfg: PersistenceConfig):
    """Return (index, act, per_stage_audit) where `index` is the stage at
    which the resetting streak counter M_k first reaches theta, or
    (None, False, audit) if the budget is exhausted first. No ground truth
    anywhere in this function's signature."""
    audit = []
    streak = 0
    for idx, stage in enumerate(stages):
        native = stage["native"]
        H = np.asarray(native["hessian"], dtype=float).reshape(6, 6)
        R = np.asarray(native["R"], dtype=float)
        t = np.asarray(native["t"], dtype=float)
        features = stage.get("features")
        d = stage_decision_with_variant(R, t, H, spec, native_cfg, pcfg, features=features)
        streak = streak + 1 if d["act"] else 0
        d["streak_M_k"] = streak
        audit.append(d)
        if streak >= pcfg.theta:
            return idx, True, audit
    return None, False, audit
