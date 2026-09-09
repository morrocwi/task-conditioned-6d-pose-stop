#!/usr/bin/env python3
"""Derive the native retained-sensitivity model's lambda_ref and cap from
TRAIN only.

Fifth real-data cycle (2026-09-09, "run5", PROP-NATIVE-01/02 --
ops/HANDOFF_2026-09-09_external_dataset.md, "Fifth cycle authorized"). This
script computes TWO numbers from `lab/data/bop_lmo_episodes_native/train.jsonl`
ONLY. It never reads calibration.jsonl or test.jsonl.

1. lambda_ref := median(lambda_min(H_k)) across all TRAIN stages (640 stage
   readings from 40 episodes x 16 stages), using H_k's real eigendecomposition
   (`lab/decay_predictor.py:eigh_H`, the same routine family already reported
   in run4). Median chosen as a predeclared, robust-to-outlier central-
   tendency choice.

2. cap := the 75th percentile of the Euclidean norm of the online-observable
   local-dispersion proxy (`lab/bop_icp_backend.py:local_proxy`, the same
   6-component proxy already carried in `observable_features`) across the
   same 640 TRAIN stage readings. p75 chosen for consistency with run2's own
   house convention (`lab/derive_tolerances_from_train.py`) for turning a
   TRAIN-only distribution into a single predeclared design constant.

Honest record of a discarded candidate CAP (kept here, not silently dropped):
an earlier hand-picked round-number candidate, CAP=0.005 (5mm / ~0.29deg,
chosen only because it sits below every task tolerance component), was
checked against CALIBRATION-only diagnostics (never test.jsonl) before this
file was frozen. It made the invariance test structurally vacuous: since CAP
was smaller than every masked task tolerance, EVERY perturbed candidate
trivially passed regardless of lambda_min, so ACT fired at stage k=0 on
effectively every episode independent of any real signal -- not a genuine
finding, a definitional flaw in the predeclared constant itself. The p75
local-dispersion-proxy-norm CAP below ties the bound to a real observed
per-stage quantity instead of an arbitrary round number, while remaining
derived from TRAIN alone -- see
`lab/results/real-bop-lmo-2026-09-09-run5/RATIONALE.md` for the full account.

C (the numerator of the perturbation-magnitude rule) is then defined as
C := cap * lambda_ref, so that m_j = min(C / lambda_j, cap) saturates at cap
exactly when lambda_j is at the TRAIN-typical (median) scale, and shrinks
below cap for better-constrained (larger lambda_j) directions -- zero free
parameters chosen by looking at CALIBRATION or FINAL TEST outcomes.

Run: `python3 lab/derive_native_threshold_from_train.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.decay_predictor import eigh_H

TRAIN_PATH = ROOT / "lab" / "data" / "bop_lmo_episodes_native" / "train.jsonl"
FEATURE_EPS = 1e-9


def main():
    rows = [json.loads(ln) for ln in TRAIN_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    lam_mins = []
    proxy_norms = []
    for ep in rows:
        for stage in ep["stages"]:
            H = np.asarray(stage["native"]["hessian"], dtype=float).reshape(6, 6)
            eigvals, _ = eigh_H(H)
            lam_mins.append(float(eigvals[0]))
            proxy_log = np.asarray(stage["features"][6:12], dtype=float)
            proxy = np.exp(proxy_log) - FEATURE_EPS
            proxy_norms.append(float(np.linalg.norm(proxy)))
    lam_mins = np.asarray(lam_mins, dtype=float)
    proxy_norms = np.asarray(proxy_norms, dtype=float)
    lambda_ref = float(np.median(lam_mins))
    cap = float(np.percentile(proxy_norms, 75))
    out = {
        "n_train_episodes": len(rows),
        "n_stage_readings": int(lam_mins.size),
        "lambda_min_summary": {
            "min": float(lam_mins.min()),
            "p25": float(np.percentile(lam_mins, 25)),
            "median": lambda_ref,
            "p75": float(np.percentile(lam_mins, 75)),
            "max": float(lam_mins.max()),
        },
        "local_dispersion_proxy_norm_summary": {
            "min": float(proxy_norms.min()),
            "p25": float(np.percentile(proxy_norms, 25)),
            "median": float(np.median(proxy_norms)),
            "p75": cap,
            "p90": float(np.percentile(proxy_norms, 90)),
            "max": float(proxy_norms.max()),
        },
        "lambda_ref": lambda_ref,
        "lambda_ref_method": "median(lambda_min(H_k)) over every TRAIN stage",
        "cap": cap,
        "cap_method": "p75(||local_dispersion_proxy||) over every TRAIN stage",
        "c_scale": cap * lambda_ref,
        "perturbation_magnitude_rule": "m_j = min(c_scale / lambda_min(H_k), cap)",
        "discarded_candidate_cap": {
            "value": 0.005,
            "reason_discarded": "CALIBRATION-only diagnostic showed this hand-picked round number sits below every task tolerance component, making the invariance test structurally vacuous (trivial ACT at k=0 on effectively every episode, independent of any real signal)",
        },
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
