#!/usr/bin/env python3
"""Executable finite completion-envelope harness.

This implements the NEW project definitions in TOLEDO_COMPLETION_ENVELOPE.md.
It does not implement or modify Toledo itself.
"""
from __future__ import annotations

import itertools
import math

PASS = "PASS"
FAIL = "FAIL"
HOLD = "HOLD"

TASKS = {
    "top_suction": {
        "kind": "box",
        "mask": (0, 1, 2, 3, 4),
        "tol": (0.005, 0.005, 0.008, math.radians(5), math.radians(5), math.inf),
    },
    "label_alignment": {
        "kind": "box",
        "mask": (0, 1, 2, 3, 4, 5),
        "tol": (0.005, 0.005, 0.008, math.radians(5), math.radians(5), math.radians(4)),
    },
    "keyed_insertion": {
        "kind": "l1",
        "mask": (0, 1, 5),
        "tol": (0.004, 0.004, math.inf, math.inf, math.inf, math.radians(3)),
    },
}


def task_reader(task: str, error):
    """Binary downstream reader on a local 6-D pose-error candidate."""
    spec = TASKS[task]
    if spec["kind"] == "box":
        ok = all(abs(float(error[i])) <= float(spec["tol"][i]) for i in spec["mask"])
    else:
        ok = sum(abs(float(error[i])) / float(spec["tol"][i]) for i in spec["mask"]) <= 1.0
    return PASS if ok else FAIL


def box_completion_set(bounds):
    """Finite retained completion set: center plus all vertices of |e_i|<=bounds_i.

    For the monotone absolute-value readers used here, center+vertices is enough to
    detect robust PASS vs ambiguity. This is not claimed for arbitrary task readers.
    """
    b = tuple(float(x) for x in bounds)
    center = (0.0,) * len(b)
    vertices = [tuple(sign * bi for sign, bi in zip(signs, b))
                for signs in itertools.product((-1.0, 1.0), repeat=len(b))]
    return [center] + vertices


def task_reader_image(task, completions):
    return {task_reader(task, z) for z in completions}


def robust_verdict(task, completions):
    image = task_reader_image(task, completions)
    if image == {PASS}:
        return PASS
    if image == {FAIL}:
        return FAIL
    return HOLD


def binary_reader_diameter(task, completions):
    """Eq. C4 for a binary 0/1 reader metric."""
    return 0 if len(task_reader_image(task, completions)) <= 1 else 1


def envelope_verdict(task, bounds):
    return robust_verdict(task, box_completion_set(bounds))


def contraction_value(task, before_bounds, after_bounds):
    """Observed finite-envelope contraction W(before)-W(after)."""
    before = binary_reader_diameter(task, box_completion_set(before_bounds))
    after = binary_reader_diameter(task, box_completion_set(after_bounds))
    return before - after


def choose_by_contraction_per_cost(task, bounds, actions):
    """Rank deterministic toy refinements by C5/COST.

    actions: iterable of (name, next_bounds, positive_cost).
    No claim is made that an external perception action actually realizes next_bounds.
    """
    ranked = []
    for name, nxt, cost in actions:
        if cost <= 0:
            raise ValueError("cost must be positive")
        gain = contraction_value(task, bounds, nxt)
        ranked.append((gain / float(cost), gain, -float(cost), name))
    return max(ranked)[-1] if ranked else None


def demo():
    common = (0.002, 0.002, 0.004, math.radians(2), math.radians(2), math.radians(12))
    yaw_refined = (*common[:5], math.radians(2))
    insertion_coupled = (0.003, 0.003, 0.0, 0.0, 0.0, math.radians(2.4))
    out = {
        "top_suction_wide_yaw": envelope_verdict("top_suction", common),
        "label_wide_yaw": envelope_verdict("label_alignment", common),
        "label_after_yaw_refinement": envelope_verdict("label_alignment", yaw_refined),
        "keyed_insertion_coupled": envelope_verdict("keyed_insertion", insertion_coupled),
    }
    return out


if __name__ == "__main__":
    for k, v in demo().items():
        print(f"{k}: {v}")
