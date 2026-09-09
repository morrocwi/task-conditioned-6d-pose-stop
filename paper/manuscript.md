# Task-Conditioned Early Stopping for Iterative 6D Pose Refinement under Incomplete State Knowledge

**Yaoharee Lahtee**  
Open Civil Science Initiative, Bangkok, Thailand

## Abstract

Iterative 6D pose pipelines usually terminate when an estimator-side convergence condition is reached or a fixed refinement budget is exhausted. A downstream manipulation task can become executable earlier. This paper studies that mismatch directly: whether pose refinement can terminate at the first stage at which current pose evidence is sufficient for the declared task, even while the estimator would continue refining.

We implement the proposal on an executable numerical 6D backend using iterative point-to-point ICP/Kabsch registration on generated asymmetric 3-D point clouds. The task gate sees only current refinement readouts and declared task tolerances; hidden ground-truth pose error is unavailable to the gate. Task thresholds are selected on a tuning split, frozen, checked on a separate safety-calibration split, and evaluated on held-out test seeds. In a frozen standard run, estimator-side stopping occurs at mean stage 14.033, whereas task-conditioned stopping occurs at mean stages 10.292, 10.550, and 11.542 for top suction, label alignment, and keyed insertion. `k_task < k_estimator` occurs in 90.0–92.5% of held-out in-distribution episodes, with completion differences of -0.83 to -1.67 percentage points and HOLD rates of 6.67–9.17%.

To extend the model beyond point uncertainty estimates, we introduce a Toledo-compatible **admissible completion set**: the retained set of latent pose completions not excluded by the current readout and declared constraints. Task-conditioned stopping is licensed only when every retained completion yields the same downstream task verdict. Thus exact latent-state knowledge is not required; reader-invariance across the unresolved set is. A finite executable harness demonstrates that wide yaw uncertainty can be irrelevant for top suction but unresolved for label alignment, and that joint insertion uncertainty can remain ambiguous even when marginal coordinate bounds individually pass.

A declared two-fold sensor-noise shift exposes the practical boundary: label-alignment completion falls substantially through HOLD, while keyed insertion returns HOLD on all stress episodes. The present evidence therefore establishes only public numerical-backend and finite completion-envelope behavior. RGB-D/GPU acceleration, physical manipulation non-inferiority, and universal latent coverage remain open.

## 1. Problem

A pose estimator answers one question:

> Has the pose estimate converged sufficiently according to the estimator's own criterion?

A manipulation controller asks another:

> Is the current pose evidence already sufficient for this task?

These conditions need not become true at the same refinement stage.

Let `(P_k,U_k)` be the pose readout and uncertainty evidence after refinement stage `k`. Let `C_E` be an estimator-side stopping rule and `C_T` a task-conditioned rule.

The computational opportunity is

\[
C_T(P_k,U_k,T,C)=PASS
\quad\text{while}\quad
C_E(P_k,U_k)=CONTINUE.
\]

Define

\[
k_T=\min\{k:C_T(P_k,U_k,T,C)=PASS\}
\]

and let `k_E` denote estimator-side stopping.

The central empirical event is

\[
\boxed{k_T<k_E.}
\]

The method is useful only if this occurs often enough to reduce total perception cost without an unacceptable loss of task completion.

## 2. Task-conditioned stopping

The first implementation uses only

```text
ACT | CONTINUE_REFINEMENT | HOLD
```

After each ordinary refinement stage, the system evaluates current observable evidence against a declared downstream task condition.

- `ACT`: current evidence licenses the task.
- `CONTINUE_REFINEMENT`: the task remains unresolved and another refinement stage is available.
- `HOLD`: execution is not licensed and no admissible continuation remains within the declared budget.

The gate must evaluate the task jointly. Independent coordinate thresholds are insufficient in general. For keyed insertion, for example, separate translation and yaw bounds may each appear acceptable while their combined error exceeds the insertion tolerance.

## 3. Incomplete knowledge: a Toledo-compatible completion envelope

This section introduces new project equations rooted in Toledo's retained-state and reader discipline. They are not existing Toledo equations.

### 3.1 Toledo anchors

**[EXISTING–TOLEDO]** retained-state stepper:

\[
S_{n+1}=F(S_n,u_n,c_n,T_n).
\]

**[EXISTING–TOLEDO]** admissible domain translation and reader preservation:

\[
q_D(F(z,u,c,T))=F_D^{\#}(q_D(z),u,c,T),
\]

\[
O_D(z;Q,c)=O_D^{\#}(q_D(z);Q,c).
\]

**[EXISTING–TOLEDO]** finite-horizon reader equivalence:

\[
z\sim_{Q,O,c,L}z'
\iff
O(F^kz)=O(F^kz')\quad\forall k\le L.
\]

The relevant consequence is that latent states need not be identical in order to be equivalent for one declared task reader.

### 3.2 Admissible completion set

**[NEW–DEFINITION]**

For current retained readout `r_k`, context `c`, and declared admissibility constraints `A`, define

\[
\boxed{
\mathcal C_k
=
\{z:\;z\text{ remains compatible with }r_k\text{ and }A(z,r_k,c)=1\}.
}
\]

`C_k` does not mean “the states that are true.” It means only “the states not yet excluded by the declared construction.”

### 3.3 Task-reader image

**[NEW–DERIVATION/PROPOSAL]**

Let `O_T` be the downstream task reader:

\[
\boxed{
\mathcal Y_{T,k}
=
\{O_T(z):z\in\mathcal C_k\}.
}
\]

For a binary task reader, `Y_T,k` is a subset of `{PASS,FAIL}`.

### 3.4 Robust task verdict without latent certainty

**[NEW–DERIVATION/PROPOSAL]**

\[
\boxed{
V_{T,k}=
\begin{cases}
PASS,&\mathcal Y_{T,k}=\{PASS\},\\
FAIL,&\mathcal Y_{T,k}=\{FAIL\},\\
HOLD,&\mathcal Y_{T,k}=\{PASS,FAIL\}.
\end{cases}
}
\]

The system therefore does not need to know the latent pose exactly before acting. It needs the unresolved alternatives to be **task-equivalent**.

This yields the stronger stopping principle

\[
\boxed{
\text{Stop when remaining uncertainty cannot change the declared task verdict.}
}
\]

rather than

\[
\text{Stop only when pose uncertainty is globally small.}
\]

### 3.5 Task ambiguity

**[NEW–DEFINITION]**

For a metric `d_T` over task readouts,

\[
\boxed{
W_T(\mathcal C_k)
=
\sup_{z,z'\in\mathcal C_k}
 d_T(O_T(z),O_T(z')).
}
\]

For a binary reader, let equal verdicts have distance zero and unequal verdicts distance one. Then

\[
W_T\in\{0,1\}.
\]

`W_T=0` says that the unresolved states are indistinguishable to this task reader. It does not say that the pose itself is known exactly.

### 3.6 Refinement value

**[NEW–DERIVATION/PROPOSAL]**

If a perception action `a` can produce one of the next completion sets in `Phi_a(C_k)`, define guaranteed task-reader contraction

\[
\boxed{
G_T(a\mid\mathcal C_k)
=
W_T(\mathcal C_k)
-
\sup_{\mathcal C'\in\Phi_a(\mathcal C_k)}W_T(\mathcal C').
}
\]

A cost-sensitive extension is

\[
\boxed{
J_T(a)=\frac{G_T(a\mid\mathcal C_k)}{c(a)},\qquad c(a)>0.
}
\]

This provides a route for extending the present binary STOP/CONTINUE system to richer future choices such as another refinement stage, a new camera view, or a broad reset. The equation does not certify the transition model `Phi_a`; that model must be tested independently.

### 3.7 Finite computational approximation

**[NEW–DEFINITION]**

A computer retains only a finite approximation

\[
\widehat{\mathcal C}_k^{(M)}
=
\{z_k^{(1)},\ldots,z_k^{(M)}\}.
\]

Therefore

\[
\boxed{
\widehat V_{T,k}^{(M)}=PASS
\not\Rightarrow
V_{T,k}=PASS
}
\]

unless the finite construction has an independently justified coverage guarantee.

This non-collapse is important: an executable imagination set is a computational readout of unresolved possibilities, not the hidden world itself.

## 4. Executable completion-envelope controls

The public repository implements the equations above in `theory/completion_envelope.py`.

The current finite envelope represents local unresolved pose error using

\[
|e_i|\le b_i
\]

and evaluates the center plus all vertices of the six-dimensional box. For the current monotone absolute-value toy readers, this is sufficient to detect robust PASS versus ambiguity; no such claim is made for arbitrary robotic tasks.

Three controls illustrate the model.

### Task switch

With small translation/tilt bounds but a wide yaw bound:

```text
top suction     -> PASS
label alignment -> HOLD
```

The latent completion set is identical. Only the task reader changes.

### Targeted resolution

Shrinking the yaw envelope while leaving the other bounds unchanged moves label alignment from

```text
HOLD -> PASS
```

and reduces binary reader diameter from one to zero.

### Coupled insertion

For keyed insertion, each active marginal bound can be below its own coordinate tolerance while the joint L1 constraint still admits failing completions. The envelope therefore returns HOLD rather than silently collapsing marginal adequacy into joint task adequacy.

## 5. Numerical 6D backend

The second evidence layer executes iterative point-to-point ICP/Kabsch registration on generated asymmetric 3-D point clouds.

The gate does not receive hidden ground-truth pose error. It receives a local linearized dispersion proxy derived from the current ICP state. This quantity is deliberately not described as a calibrated posterior or confidence interval.

A task-specific stopping threshold is:

1. selected on a tuning split;
2. frozen;
3. checked on a disjoint safety-calibration split;
4. evaluated on held-out test seeds.

The same underlying trajectories are evaluated under four stopping policies:

```text
fixed-8 | full | estimator-stop | task-stop
```

for

```text
top suction | label alignment | keyed insertion.
```

## 6. Frozen standard-profile result

The standard profile contains 96 tuning episodes, 96 independent safety-calibration episodes, 120 held-out in-distribution episodes across three test seeds, and 60 held-out two-fold sensor-noise stress episodes.

| Task | estimator mean k | task mean k | `k_T < k_E` | estimator completion | task completion | task HOLD |
|---|---:|---:|---:|---:|---:|---:|
| top suction | 14.033 | 10.292 | 92.50% | 94.17% | 93.33% | 6.67% |
| label alignment | 14.033 | 10.550 | 90.83% | 93.33% | 91.67% | 6.67% |
| keyed insertion | 14.033 | 11.542 | 90.00% | 92.50% | 90.83% | 9.17% |

The paired bootstrap intervals for `k_T-k_E` are below zero for all three in-distribution tasks in the frozen run.

These data support the numerical statement that task-conditioned stopping can occur earlier than the selected estimator criterion in the declared registration fixture.

They do not establish real-camera acceleration.

## 7. Failure boundary

The stopping thresholds are frozen and then evaluated under a two-fold observation-noise shift.

Top suction remains largely usable. Label alignment loses substantial completion through HOLD. Keyed insertion returns HOLD on every stress episode.

This failure is part of the result.

A selective system can lose operational value in two directions:

1. **overconfidence**: ACT occurs on hidden failing states;
2. **over-conservatism**: the completion envelope or calibrated gate becomes too broad and returns HOLD too often.

The objective is therefore not “stop as early as possible.” It is to find the earliest task-equivalent state without confusing unresolved latent alternatives for irrelevant ones.

## 8. Evidence boundary

The current public record supports:

- executable task-conditioned stopping on a numerical iterative 6D registration backend;
- disjoint tuning, safety-calibration, and test splits;
- held-out comparison against estimator-side stopping;
- explicit ACT, HOLD, unsafe-ACT, completion, and iteration reporting;
- executable finite completion-envelope controls for incomplete latent knowledge;
- automated public CI checks.

The record does not yet support:

- real RGB-D/GPU acceleration;
- FoundationPose-specific speedup;
- physical manipulation non-inferiority;
- safety certification;
- universal coverage of the finite completion set;
- an IDM-specific advantage.

## 9. Conclusion

Estimator convergence and task sufficiency are different stopping conditions.

The first version of this project asked whether a task can become ready before a pose estimator is finished. The numerical backend now shows that this event is executable and common in the declared in-distribution fixture.

The completion-envelope extension strengthens the question further. Exact latent-state certainty is not required if every still-admissible latent completion belongs to the same task-reader equivalence class.

The resulting principle is

\[
\boxed{
\text{Do not require complete pose knowledge when the unresolved distinctions cannot change the task.}
}
\]

but with an equally important constraint:

\[
\boxed{
\text{Finite imagined completions are evidence only to the extent that their coverage is independently justified.}
}
\]

The next empirical step is a real iterative RGB-D pose backend in which the same STOP/CONTINUE/HOLD rule and completion-envelope logic can be evaluated against measured task outcome and end-to-end latency.
