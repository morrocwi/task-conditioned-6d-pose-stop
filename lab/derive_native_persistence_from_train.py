#!/usr/bin/env python3
"""Derive PROP-NATIVE-03's persistence threshold theta (path a) and the
residual-scale reference/cap-multiplier (path b) from TRAIN only.

Sixth real-data cycle. Reads ONLY
`lab/data/bop_lmo_episodes_native/train.jsonl` (the identical file runs 5/6
share -- no new data generation for run 6). Never reads calibration.jsonl or
test.jsonl. No ground truth (`oracle`) is read here at all -- theta and the
residual-scale reference are derived purely from the ONLINE-observable
per-stage invariant sequence and feature vector, consistent with PROP-
NATIVE-01's own no-T*-anywhere discipline (this script does not even use T*
offline, unlike e.g. run2's tolerance derivation, because theta/residual-
scale-ref are themselves part of the online decision rule, not an external
task-tolerance choice).

Diagnostic (reported honestly, predicted BEFORE opening test.jsonl):
PROP-NATIVE-02's single-instant invariance check (`lab/native_sensitivity.py
:stage_decision`, called with run5's frozen lambda_ref/cap, reused
byte-identical) evaluates True at EVERY one of the 640 TRAIN stage readings
(40 episodes x 16 stages), for all three tasks. The initial streak length is
therefore 16/16 for every TRAIN episode -- there is no variation in the
per-stage indicator on TRAIN to threshold against. theta is chosen from a
principled, non-arbitrary rule below despite this; the run-6 predeclaration
states plainly that this structural finding predicts persistence ALONE
(path a) will simply convert into a fixed-stage stopping rule (ACT
deterministically at k=theta-1 on this TRAIN evidence), not a genuine
noise filter -- exactly the honest, disclosed prediction the handoff asked
for before running against test.jsonl.

theta (path a): set to 4, chosen as the smallest reused-and-frozen system
constant already established in a DIFFERENT run's predeclaration (run 3's
Bonferroni checkpoint set {4,8,12,15}, `lab/results/real-bop-lmo-2026-09-09-
run3/config.json`) -- avoids inventing a new arbitrary round number and ties
the choice to a value this repo already froze for an unrelated reason before
run 6 was ever conceived, while still being small enough to "isolate the
memory variable cleanly" per the handoff's own preference for path (a).

residual_scale_ref / residual_cap_multiplier (path b, only used by the
disclosed follow-up variant): residual_scale_ref := median(ICP residual
RMSE) over all 640 TRAIN stage readings (features[1] = log RMSE); the same
median-based, robust-to-outlier convention run5's own lambda_ref derivation
used. residual_cap_multiplier := 2.0 (a predeclared doubling of run5's own
CAP, the smallest change that lets the residual-scale factor actually alter
the bound rather than being clipped back down to the unmodified CAP on
every stage where the multiplier could exceed 1x) -- both frozen here,
before test.jsonl is read for run 6.

Run: `python3 lab/derive_native_persistence_from_train.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.native_sensitivity import NativeConfig, stage_decision

TRAIN_PATH = ROOT / "lab" / "data" / "bop_lmo_episodes_native" / "train.jsonl"
TASKS_PATH = ROOT / "lab" / "results" / "real-bop-lmo-2026-09-09-run5" / "config.json"

# Reused byte-identical from run5's own TRAIN-derived native-sensitivity config.
LAMBDA_REF = 0.33927484832749355
CAP = 0.03274911721483188
THETA = 4
RESIDUAL_CAP_MULTIPLIER = 2.0


def main():
    rows = [json.loads(ln) for ln in TRAIN_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))["tasks"]
    native_cfg = NativeConfig(lambda_ref=LAMBDA_REF, cap=CAP)

    residual_rmses = []
    for ep in rows:
        for stage in ep["stages"]:
            residual_rmses.append(float(np.exp(float(stage["features"][1])) - 1e-9))
    residual_rmses = np.asarray(residual_rmses, dtype=float)
    residual_scale_ref = float(np.median(residual_rmses))

    per_task = {}
    for tname, spec in tasks.items():
        initial_streaks = []
        stage_true_rate = np.zeros(16)
        for ep in rows:
            seq = []
            for stage in ep["stages"]:
                native = stage["native"]
                H = np.asarray(native["hessian"], dtype=float).reshape(6, 6)
                R = np.asarray(native["R"], dtype=float)
                t = np.asarray(native["t"], dtype=float)
                d = stage_decision(R, t, H, spec, native_cfg)
                seq.append(d["act"])
            seq = np.asarray(seq)
            stage_true_rate += seq.astype(int)
            i = 0
            while i < len(seq) and seq[i]:
                i += 1
            initial_streaks.append(i)
        stage_true_rate /= len(rows)
        per_task[tname] = {
            "stage_invariant_true_rate": [float(x) for x in stage_true_rate],
            "initial_streak_min": int(min(initial_streaks)),
            "initial_streak_median": float(np.median(initial_streaks)),
            "initial_streak_max": int(max(initial_streaks)),
            "fraction_full_length_streak": float(np.mean([s >= 16 for s in initial_streaks])),
        }

    out = {
        "n_train_episodes": len(rows),
        "n_stage_readings": int(residual_rmses.size),
        "reused_from_run5": {"lambda_ref": LAMBDA_REF, "cap": CAP},
        "per_task_train_diagnostic": per_task,
        "theta": THETA,
        "theta_method": "smallest of run3's own frozen Bonferroni checkpoint set {4,8,12,15} (an existing, non-run6-tuned system constant), chosen deliberately small per the handoff's stated preference to isolate the memory variable cleanly under path (a)",
        "residual_rmse_summary": {
            "min": float(residual_rmses.min()),
            "p25": float(np.percentile(residual_rmses, 25)),
            "median": residual_scale_ref,
            "p75": float(np.percentile(residual_rmses, 75)),
            "max": float(residual_rmses.max()),
        },
        "residual_scale_ref": residual_scale_ref,
        "residual_scale_ref_method": "median(ICP residual RMSE) over every TRAIN stage reading",
        "residual_cap_multiplier": RESIDUAL_CAP_MULTIPLIER,
        "residual_cap_multiplier_method": "predeclared 2x doubling of run5's own CAP, frozen before test.jsonl is read for run6",
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
