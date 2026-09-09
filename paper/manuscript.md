# Task-Conditioned Early Stopping for Iterative 6D Pose Refinement in Robotic Manipulation

**Yaoharee Lahtee**  
Open Civil Science Initiative, Bangkok, Thailand

## Abstract

Iterative 6D object-pose estimation improves pose accuracy by repeatedly refining an initial estimate before robotic manipulation. Estimator-side stopping naturally terminates this process when the estimate has sufficiently stabilized. Downstream task sufficiency is a different condition: a manipulation task may already be executable while some pose components remain uncertain and refinable.

We study task-conditioned early stopping, in which intermediate pose evidence is evaluated against a declared manipulation-success condition after each refinement stage. The same intermediate pose may therefore trigger ACT for one task and CONTINUE for another. An executable finite diagnostic with suction, orientation-alignment, and insertion predicates shows that task-conditioned policies can reduce modeled perception-action cost when uncertainty is correctly bounded, but also exposes a critical failure mode: deliberate uncertainty undercoverage reduces selective insertion success to 57.34% compared with 90.41% for a broad full-refinement baseline. These are synthetic finite-diagnostic results, not measurements of a vision model or robot.

The real-backend hypothesis is narrower: task sufficiency may occur at an earlier refinement stage than estimator-side convergence. The decisive test is whether `k_T < k_E` occurs reliably, reduces complete perception latency after gate overhead, and preserves downstream task completion within a predeclared margin.

## 1. Problem

A 6D pose estimator answers how well the object's pose is currently estimated. A robot controller needs to know whether the current estimate is sufficient for its action. These need not become true at the same refinement stage.

Let `(P_k, U_k)` be the pose and uncertainty after refinement stage `k`. Let `C_E` be an estimator-side stopping rule and `C_T` a task-conditioned sufficiency rule.

The opportunity studied here is:

```text
C_T(P_k, U_k, task, context) = PASS
while
C_E(P_k, U_k) = CONTINUE
```

Define `k_T` as the first task-sufficient stage and `k_E` as the estimator-side stopping stage. The method is useful only when `k_T < k_E` occurs often enough to offset the cost of uncertainty estimation and task gating.

## 2. Method

The first real implementation uses only three decisions:

```text
ACT | CONTINUE_REFINEMENT | HOLD
```

After each ordinary backend refinement stage, the system computes a calibrated uncertainty representation and evaluates joint task admissibility. It stops only if current evidence licenses the downstream task. It returns HOLD if execution is not licensed and continuation is unavailable or exceeds the declared budget.

The gate must consider coupled task geometry. Independent per-axis thresholds are insufficient in general: the synthetic insertion counterexample uses `x=0.7` and `yaw=0.7`, which pass separate unit thresholds but violate `abs(x)+abs(y)+abs(yaw)<=1`.

## 3. Finite diagnostic

The current repository exhausts 64 binary uncertainty fixture states for three toy tasks and two action menus. It compares one-shot, full refinement, task-mask greedy, cost-aware greedy, Dijkstra, and a pinned IDM min-plus core. Dijkstra and IDM are checked against exhaustive action-subset enumeration.

For rich-menu insertion, mean modeled costs are 15.719 for full refinement and 5.000 for cost-aware greedy, Dijkstra, and IDM. Under bounded synthetic uncertainty, these gated policies achieve the constructed 100% success condition. Under deliberate 6x undercoverage, insertion success is 90.41% for full refinement and 57.34% for the selective optimal routes.

The modeled costs are dimensionless. They are not measured GPU time.

## 4. Interpretation

The synthetic result supports the stopping logic only conditionally. Selective stopping can avoid work when the stopping evidence is reliable. When uncertainty is overly optimistic, selective stopping can preserve hidden error that broad refinement would correct.

Therefore the operational rule is:

```text
stop at task sufficiency only when the evidence for task sufficiency is sufficiently calibrated
```

## 5. Real experiment

Use one iterative 6D pose backend and one independently measurable manipulation task. Compare fixed/full refinement, estimator-side early stopping, and task-conditioned early stopping from the same initial inference.

Measure the full pipeline: initial inference, uncertainty adapter, gate, each refinement stage, and total latency. Report completion over all assigned episodes, success conditional on ACT, HOLD rate, unsafe ACT rate, and the distribution of `k_T - k_E`.

A real speed claim is supported only if task-conditioned stopping reduces total latency, not merely refinement count.

## 6. Current claim boundary

The repository does not currently establish real 6D pose acceleration, physical manipulation non-inferiority, or an IDM-specific advantage. Those claims remain open until a real backend is executed.
