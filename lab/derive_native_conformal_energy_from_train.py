#!/usr/bin/env python3
"""Predeclaration diagnostic for run 7 (PROP-NATIVE-04) -- TRAIN + CALIBRATION
only, NEVER test.jsonl.

Per this project's own standing discipline (runs 5-6's derive_* scripts) and
the run-7 plan's explicit instruction, this script must be run and its
output committed BEFORE `lab/run_native_conformal_energy.py` is ever invoked
against `lab/data/bop_lmo_episodes_native/test.jsonl`.

What is legitimately touched here, and why
-------------------------------------------
- TRAIN (`train.jsonl`): fits the log-linear error-shape model
  (`lab.run_real_system.fit_model`, unchanged) and supplies the per-stage
  `H_k` eigenvalue distribution used below.
- CALIBRATION (`calibration.jsonl`): calibrates PROP-CONF-03's per-checkpoint
  Bonferroni conformal quantile q_k (`lab.multicheckpoint.calibrate_checkpoints`,
  unchanged). This is NOT a violation of the TRAIN/CALIBRATION/TEST
  discipline -- calibrating q against the calibration split, before any test
  episode is read, is exactly the standard split-conformal recipe runs 1-3
  already used and already predeclared; only rho/theta (new free parameters
  PROP-NATIVE-04 introduces) must additionally be fixed from TRAIN alone,
  per the Toledo entry's honest_caveats risk 5, and that is done below using
  ONLY the TRAIN-computed m_k trajectory, never calibration/test outcomes.
- test.jsonl is NEVER opened by this script.

Two structural questions this diagnostic answers before test.jsonl is opened
------------------------------------------------------------------------------
1. Does eps_{k,j} = min(q_k/lambda_j, CEILING) actually differ from run 5/6's
   plain CEILING (CAP=0.03274911721483188, reused byte-identical), or does
   the ceiling term dominate the min() at essentially every stage --
   reproducing run 5/6's undersized-magnitude failure under a new name?
2. Does gate (b) -- the checkpoint's own PROP-CONF-03 certificate condition
   (`lab.native_conformal_energy.certificate_holds_at_checkpoint`) -- ever
   hold on TRAIN data at any of the four predeclared checkpoints? Run 3
   already found certificate_rate=0.0 (100% HOLD) on this exact backend/
   dataset/model/checkpoints/alpha using the SAME calibrated envelope
   machinery; if that finding reproduces on TRAIN here, it predicts, before
   test.jsonl is opened, that PROP-NATIVE-04's AND-gate will make ACT
   structurally near-impossible regardless of the native/Keystone/decay
   half of the construction -- the opposite failure mode from runs 5-6
   (over-conservatism instead of unsafe-early-ACT), exactly the Toledo
   entry's disclosed "Open risk 1".

rho, theta predeclaration (TRAIN only)
----------------------------------------
- rho = 0.5: predeclared, disclosed as the simplest non-extreme decay rate
  -- avoids both rho=0 (no memory, reduces to a single-instant check like
  run 5) and rho=1 (infinite memory, reduces to run 6's non-decaying hard
  sum). Not tuned by looking at any TRAIN, calibration, or test outcome;
  chosen before this diagnostic was run.
- theta = median, over the 40 TRAIN episodes, of the checkpoint-indexed
  accumulator m_k evaluated at the LAST predeclared checkpoint (k=15),
  using rho=0.5 above and the SAME eps/Gamma/iota construction that will
  run online at test time. Median chosen so that, on TRAIN, roughly half of
  episodes would clear the threshold by the final checkpoint if this were
  test data (the same "TRAIN-distribution-derived, not hand-picked" pattern
  runs 2/5/6 already used for tolerances/CAP/theta) -- disclosed here BEFORE
  test.jsonl is opened, never adjusted afterward.

Run: `python3 lab/derive_native_conformal_energy_from_train.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.multicheckpoint import calibrate_checkpoints, validate_checkpoints
from lab.native_conformal_energy import (
    ConformalEnergyConfig,
    certificate_holds_at_checkpoint,
    stage_checkpoint_signal,
)
from lab.native_sensitivity import NativeConfig
from lab.run_real_system import fit_model

DATA_DIR = ROOT / "lab" / "data" / "bop_lmo_episodes_native"
CHECKPOINTS = (4, 8, 12, 15)
ALPHA = 0.1  # reused byte-identical from run3 (n=40, alpha=0.1 -> K'<=4 feasible)
CEILING = 0.03274911721483188  # "weld/M.40.v1 ceiling", byte-identical run5/6 CAP
ERROR_FLOOR = [1e-05, 1e-05, 1e-05, 1e-04, 1e-04, 1e-04]  # byte-identical runs 1-6
RHO = 0.5


def load_jsonl(path):
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


TASKS = {
    "top_suction": {"kind": "box", "mask": [0, 1, 2, 3, 4],
                     "tol": [0.014, 0.013, 0.013, 0.12217304763960307, 0.13962634015954636, 1e9]},
    "label_alignment": {"kind": "box", "mask": [0, 1, 2, 3, 4, 5],
                         "tol": [0.014, 0.013, 0.013, 0.12217304763960307, 0.13962634015954636, 0.15707963267948966]},
    "keyed_insertion": {"kind": "l1", "mask": [0, 1, 5],
                         "tol": [0.014, 0.013, 1e9, 1e9, 1e9, 0.15707963267948966]},
}


def main():
    train = load_jsonl(DATA_DIR / "train.jsonl")
    cal = load_jsonl(DATA_DIR / "calibration.jsonl")
    validate_checkpoints(list(CHECKPOINTS), train[0]["stages"])

    model = fit_model(train, ERROR_FLOOR)
    q_by_checkpoint = calibrate_checkpoints(cal, model, list(CHECKPOINTS), ALPHA)
    q_report = {int(k): (None if q == float("inf") else q, rank) for k, (q, rank) in q_by_checkpoint.items()}

    cfg_template = {"checkpoints": CHECKPOINTS, "ceiling": CEILING, "rho": RHO}

    report = {"q_by_checkpoint": q_report, "ceiling": CEILING, "rho": RHO, "tasks": {}}
    stages_by_k_all = None
    for name, spec in TASKS.items():
        cfg = ConformalEnergyConfig(checkpoints=CHECKPOINTS, ceiling=CEILING, rho=RHO, theta=float("inf"))
        m_at_last_checkpoint = []
        eps_samples = []  # (checkpoint, lambda_min, eps, ceiling_bound)
        gate_b_true_count = {k: 0 for k in CHECKPOINTS}
        for ep in train:
            stages_by_k = {int(s["k"]): (idx, s) for idx, s in enumerate(ep["stages"])}
            m = 0.0
            for k_m in CHECKPOINTS:
                idx, stage = stages_by_k[k_m]
                native = stage["native"]
                H = np.asarray(native["hessian"], dtype=float).reshape(6, 6)
                R = np.asarray(native["R"], dtype=float)
                t = np.asarray(native["t"], dtype=float)
                q_checkpoint, _rank = q_by_checkpoint[k_m]
                sig = stage_checkpoint_signal(R, t, H, spec, q_checkpoint, cfg)
                m = RHO * m + (sig["gamma_k"] if sig["iota_k"] else 0.0)
                eps_samples.append((k_m, sig["lambda_min"], sig["eps"], sig["eps"] >= CEILING - 1e-12))
                if certificate_holds_at_checkpoint(stage, model, q_checkpoint, spec):
                    gate_b_true_count[k_m] += 1
            m_at_last_checkpoint.append(m)
        theta = float(np.median(m_at_last_checkpoint))

        eps_arr = np.asarray([e[2] for e in eps_samples], dtype=float)
        ceiling_bound_frac = float(np.mean([e[3] for e in eps_samples]))
        report["tasks"][name] = {
            "theta_predeclared_median_m_at_k15": theta,
            "m_at_k15_train_distribution": {
                "min": float(np.min(m_at_last_checkpoint)),
                "median": theta,
                "max": float(np.max(m_at_last_checkpoint)),
            },
            "eps_saturates_at_ceiling_fraction_of_stage_checkpoint_pairs": ceiling_bound_frac,
            "eps_distribution": {
                "min": float(np.min(eps_arr)), "median": float(np.median(eps_arr)), "max": float(np.max(eps_arr)),
            },
            "gate_b_certificate_true_rate_by_checkpoint": {
                int(k): gate_b_true_count[k] / len(train) for k in CHECKPOINTS
            },
        }

    out = ROOT / "lab" / "results" / "real-bop-lmo-2026-09-09-run7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "native_conformal_energy_threshold_derivation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
