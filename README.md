# Task-Conditioned Early Stopping for Iterative 6D Pose Refinement

Public research repository for an implementation-first study of **task-conditioned early stopping** in iterative 6D object-pose refinement for robotic manipulation.

## Research question

> Can iterative 6D pose refinement stop **before estimator-side convergence** because the downstream manipulation task is already executable?

The key distinction is:

```text
estimator convergence != downstream task sufficiency
```

The proposed stopping rule evaluates the current pose evidence against the declared task:

```text
pose + uncertainty + task + context
    -> ACT | CONTINUE | HOLD
```

The project does **not** claim that task-aware perception, early stopping, or pose uncertainty are new in general. The narrow empirical hypothesis is that a real iterative 6D pose backend may sometimes satisfy the downstream task at an earlier refinement stage than its own generic stopping criterion.

## Current evidence status

The repository currently contains a **finite synthetic diagnostic**, not a vision or robot performance benchmark.

Executed so far:

- 3 synthetic tasks: suction, label alignment, insertion
- 2 synthetic perception-action menus
- all 64 binary uncertainty fixture states
- 6 policies
- 2,304 policy rollouts
- paired residual-error evaluation
- bounded uncertainty and deliberate 6x undercoverage stress
- standard Dijkstra and a pinned IDM min-plus core checked against exhaustive finite enumeration

Not yet executed:

- a real 6D vision backend
- RGB-D/GPU refinement timing
- rigid-body/contact physics
- camera motion
- physical robot manipulation

## Main diagnostic result

For the rich-menu insertion fixture:

| Method | Mean modeled cost | Success: bounded | Success: undercoverage |
|---|---:|---:|---:|
| one-shot | 0.000 | 30.26% | 1.16% |
| full refinement | 15.719 | 100.00% | 90.41% |
| mask greedy | 6.000 | 100.00% | 39.14% |
| cost-aware greedy | 5.000 | 100.00% | 57.34% |
| Dijkstra | 5.000 | 100.00% | 57.34% |
| IDM min-plus | 5.000 | 100.00% | 57.34% |

The modeled cost reduction is **not a measured GPU speedup**. The negative result is equally important: when uncertainty undercovers true error, selective stopping can preserve hidden pose error that broad refinement would correct.

## Run

Python 3.10+; no third-party dependencies required for the finite fixture.

```bash
python benchmark.py
python -m unittest discover -s tests -v
```

## Next implementation gate

The next experiment deliberately uses the smallest real action space:

```text
ACT | CONTINUE_REFINEMENT | HOLD
```

Compare:

```text
fixed/full refinement
vs estimator-side early stopping
vs task-conditioned early stopping
```

The decisive event is `k_task < k_estimator`: the task becomes ready before the estimator reaches its generic stopping condition.

## IDM

IDM is retained only as an optional finite routing backend for richer future action menus. It is **not** the novelty claim. In the current small graph fixture, Dijkstra and the IDM min-plus core recover the same modeled optimum, while Dijkstra has lower routing overhead.

## Reproducibility

Synthetic benchmark seed: `20260909`.

## Author

Yaoharee Lahtee  
Open Civil Science Initiative, Bangkok, Thailand
