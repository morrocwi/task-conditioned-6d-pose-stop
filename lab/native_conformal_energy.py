"""Conformal-scaled, Keystone-energy, decay-accumulated task admissibility
(PROP-NATIVE-04).

Seventh real-data cycle (ops/HANDOFF_2026-09-09_external_dataset.md, run 7
authorization following the 2026-09-09 ultracode team meeting convened after
run 6 (PROP-NATIVE-03) was refuted). Toledo proposal:
~/ANSE.ASIA/toledo/registry/proposals/native_retained_sensitivity.json,
id "PROP-NATIVE-04".

This is a targeted composition of THREE fixes, one per diagnosed defect from
runs 5-6, layered on top of code that is reused, never reimplemented:

1. Perturbation magnitude scale (fixes run 5's diagnosed defect: a fixed
   TRAIN-population CAP with no absolute-error-scale information):
       eps_{k,j} := min(q_k / lambda_j, CEILING)
   `q_k` is PROP-CONF-03's per-checkpoint Bonferroni conformal quantile
   (`cqts.safety.safe_multi_checkpoint_quantiles`, reused unchanged, via
   `lab/multicheckpoint.py:calibrate_checkpoints`, also reused unchanged),
   calibrated OFFLINE on CALIBRATION only, never against a test episode.
   `lambda_j` is `H_k`'s single smallest eigenvalue (the same "least
   constrained direction" convention PROP-NATIVE-02/03 already used --
   `lab/native_sensitivity.py:select_least_constrained_direction`, reused
   unchanged; "for every data-justified-uncertain j" in the registered
   statement is read here as the SAME single-direction predeclared rule
   runs 5-6 already committed to, disclosed explicitly rather than silently
   generalized to a new, undisclosed multi-direction selection criterion).
   CEILING is "weld/M.40.v1 ceiling" -- reused BYTE-IDENTICAL as run 5/6's own
   CAP (`NativeConfig.cap`, from `lab/derive_native_threshold_from_train.py`),
   per the Toledo entry's own honest_caveats ("Open risk 4 ... the same
   UNPROVEN bridge PROP-NATIVE-02 already carried"); this proposal does not
   mint a new numeric ceiling, it reuses the existing registered one.

2. Per-stage signal (fixes run 6's diagnosed defect: a non-toggling boolean):
       Gamma_k := lambda_j * eps_{k,j}^2
   the Dirichlet quadratic form Phi^T L_R Phi = I(Phi) (Toledo Keystone,
   `formal/IDM_Keystone.v`), specialized L_R -> H_k, Phi -> eps_k, restricted
   to the single chosen direction j (consistent with (1) above) so this is a
   real scalar energy value, not a 0/1 flag -- even when the invariance flag
   iota_k never toggles, Gamma_k still varies stage to stage with H_k's own
   spectrum and the calibrated magnitude.

3. Accumulator (fixes run 6's diagnosed defect: a hard reset-to-zero streak):
       m_k := rho * m_{k-1} + iota_k * Gamma_k,   m_0 := 0,   rho in (0,1)
   a decayed real-valued running sum (A2-FOLD, real-valued carrier) instead
   of a boolean streak counter, so isolated non-invariant stages do not wipe
   accumulated evidence to zero, and stale evidence decays instead of
   persisting forever.

Checkpoint-indexed accumulator -- disclosed interpretation choice
------------------------------------------------------------------
`q_k` (and therefore `eps_{k,j}` and `Gamma_k`) is only a well-defined
quantity AT the K'=4 predeclared Bonferroni checkpoints (reused from run 3:
checkpoints=[4,8,12,15], alpha=0.1) -- PROP-CONF-03's per-checkpoint
regression-bias correction is not calibrated for arbitrary intermediate ICP
iterations. The registered statement's index `k` is therefore read here,
disclosed explicitly (mirroring `lab/native_persistence.py`'s own disclosed
streak-vs-sum interpretation call), as ranging over the predeclared
checkpoint sequence k_1=4, k_2=8, k_3=12, k_4=15 -- NOT every ICP iteration.
`m_k` accumulates once per checkpoint, in order, not once per ICP stage.

Decision -- disclosed reading of "q_{k_m} <= tau_i"
-----------------------------------------------------
The registered LaTeX writes the second ACT condition as a scalar comparison
`q_{k_m} <= tau_i`. Read literally, this compares a dimensionless log-space
nonconformity quantile (q_k, order ~2 on this dataset -- see
`derive_native_conformal_energy_from_train.py`'s TRAIN diagnostic) against a
physical task tolerance tau_i (metres / radians) -- a units mismatch, not a
sensible inequality as literally typeset. This module reads the intended
condition, disclosed explicitly rather than silently patched, as PROP-CONF-03's
OWN already-implemented, physically well-typed certificate condition at that
checkpoint: the calibrated completion-error bound built from q_{k_m}
(`lab.run_real_system.completion_bounds`) already lies inside the task's
admissible region (`lab.run_real_system.task_pass`, i.e. the same condition
`lab/multicheckpoint.py:first_certificate_checkpoint` already evaluates,
reused unchanged here). This ties the decision back to the project's own
declared task-admissible region C11, exactly as the honest_caveats intend,
without a dimensionally invalid literal comparison.

ACT <=> the first predeclared checkpoint k_m (in ascending order) such that
BOTH (a) m_{k_m} >= theta and (b) the checkpoint's own PROP-CONF-03
certificate condition holds.

No ground truth is used anywhere in this module. Oracle abs_pose_error_6d is
read only afterward by the evaluator (`lab/run_native_conformal_energy.py`),
in the same offline-oracle role T* already plays for runs 1-6.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqts.safety import fail_closed_task_pass
from lab.native_sensitivity import (
    NativeConfig,
    deviation_axes,
    perturbed_pose,
    select_least_constrained_direction,
)
from lab.run_real_system import completion_bounds, task_pass


@dataclass(frozen=True)
class ConformalEnergyConfig:
    checkpoints: tuple  # predeclared checkpoint stage k values, ascending, e.g. (4, 8, 12, 15)
    ceiling: float       # "weld/M.40.v1 ceiling", reused byte-identical from run5/6's CAP
    rho: float           # decay rate, predeclared from TRAIN only, in (0, 1)
    theta: float         # accumulation threshold, predeclared from TRAIN only


def perturbation_magnitude_conformal(lambda_min: float, q_checkpoint: float, cfg: ConformalEnergyConfig):
    """eps_{k,j} = min(q_k/lambda_j, CEILING); fail-closed to CEILING (never
    unbounded, never silently zero) if lambda_min is non-finite or <= 0,
    mirroring `lab.native_sensitivity.perturbation_magnitude`'s fail-closed
    convention exactly."""
    import math
    if not math.isfinite(lambda_min) or lambda_min <= 0.0:
        return float(cfg.ceiling), True
    if not math.isfinite(q_checkpoint) or q_checkpoint < 0.0:
        return float(cfg.ceiling), True
    m = min(q_checkpoint / lambda_min, cfg.ceiling)
    if not math.isfinite(m) or m < 0.0:
        return float(cfg.ceiling), True
    return float(m), False


def stage_checkpoint_signal(R, t, H, spec: dict, q_checkpoint: float, cfg: ConformalEnergyConfig) -> dict:
    """Compute (eps, degenerate, iota_k, Gamma_k) at one predeclared
    checkpoint for one task spec, reusing PROP-NATIVE-02's own eigen-
    extraction / perturbation-composition / task-reader machinery
    unmodified. No ground truth anywhere in this function's signature."""
    lambda_min, v_min = select_least_constrained_direction(H)
    eps, degenerate = perturbation_magnitude_conformal(lambda_min, q_checkpoint, cfg)

    baseline_dev = np.zeros(6, dtype=float)
    verdict_baseline = bool(fail_closed_task_pass(spec, baseline_dev))
    R_plus, t_plus = perturbed_pose(R, t, v_min, eps, +1.0)
    R_minus, t_minus = perturbed_pose(R, t, v_min, eps, -1.0)
    dev_plus = deviation_axes(R, t, R_plus, t_plus)
    dev_minus = deviation_axes(R, t, R_minus, t_minus)
    verdict_plus = bool(fail_closed_task_pass(spec, dev_plus))
    verdict_minus = bool(fail_closed_task_pass(spec, dev_minus))
    iota_k = bool(verdict_baseline and verdict_plus and verdict_minus)
    gamma_k = float(lambda_min * eps * eps) if np.isfinite(lambda_min) and lambda_min > 0 else 0.0

    return {
        "lambda_min": float(lambda_min),
        "eps": float(eps),
        "degenerate": bool(degenerate),
        "iota_k": iota_k,
        "gamma_k": gamma_k,
        "verdict_baseline": verdict_baseline,
        "verdict_plus": verdict_plus,
        "verdict_minus": verdict_minus,
    }


def certificate_holds_at_checkpoint(stage, model, q_checkpoint, spec) -> bool:
    """Disclosed reading of the registered condition 'q_{k_m} <= tau_i':
    PROP-CONF-03's own certificate condition at that checkpoint (reused
    unchanged from `lab.run_real_system.completion_bounds` / `task_pass`,
    the same function `lab/multicheckpoint.py:first_certificate_checkpoint`
    already calls). See module docstring for why the literal scalar
    comparison is a units mismatch and this is the physically well-typed
    stand-in."""
    bounds = completion_bounds(stage, model, q_checkpoint)
    return bool(task_pass(spec, bounds))


def first_act_checkpoint(ep, spec: dict, model, q_by_checkpoint: dict, native_cfg: NativeConfig,
                          cfg: ConformalEnergyConfig):
    """Return (stage_index, acted, per_checkpoint_audit) -- the stage index
    (in ep['stages']) of the first predeclared checkpoint at which BOTH the
    decayed Keystone-energy accumulator m_{k_m} >= theta AND the checkpoint's
    own PROP-CONF-03 certificate condition hold, or (None, False, audit) if
    no checkpoint in the predeclared list satisfies both. Ground truth is
    absent from this function's signature."""
    m = 0.0
    audit = []
    stages_by_k = {int(s["k"]): (idx, s) for idx, s in enumerate(ep["stages"])}
    for k_m in cfg.checkpoints:
        if k_m not in stages_by_k:
            raise ValueError(f"{ep['episode_id']}: predeclared checkpoint k={k_m} not found in stages")
        idx, stage = stages_by_k[k_m]
        native = stage["native"]
        H = np.asarray(native["hessian"], dtype=float).reshape(6, 6)
        R = np.asarray(native["R"], dtype=float)
        t = np.asarray(native["t"], dtype=float)
        q_checkpoint, _rank = q_by_checkpoint[k_m]
        sig = stage_checkpoint_signal(R, t, H, spec, q_checkpoint, cfg)
        m = cfg.rho * m + (sig["gamma_k"] if sig["iota_k"] else 0.0)
        gate_b = certificate_holds_at_checkpoint(stage, model, q_checkpoint, spec)
        record = dict(sig)
        record.update({"k": k_m, "stage_index": idx, "m_k": float(m), "gate_b_certificate": gate_b,
                       "q_checkpoint": None if q_checkpoint == float("inf") else q_checkpoint})
        audit.append(record)
        if m >= cfg.theta and gate_b:
            return idx, True, audit
    return None, False, audit
