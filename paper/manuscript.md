# Coverage-Qualified Task Stopping for Iterative 6D Pose Refinement

**Yaoharee Lahtee**  
Open Civil Science Initiative, Bangkok, Thailand

Version 0.5 — post-adversarial-review executable numerical manuscript, 9 September 2026

## Abstract

Iterative 6D pose pipelines normally stop at a fixed refinement budget or an estimator-side convergence criterion. A downstream task may become insensitive to the unresolved pose alternatives earlier. We study one narrow question: **can refinement stop when a calibrated set of pose errors is already entirely admissible for the declared downstream task, even though the estimator would continue?**

We formalize the first such stage as a certificate hitting time. An observable error-shape model is fitted on TRAIN episodes; a single whole-trajectory split-conformal score is calibrated on disjoint CALIBRATION episodes; and the resulting pose-error completion envelope is tested against a robust task-admissibility reader. The online gate sees no hidden ground truth. If calibration is too small for the requested finite-sample conformal rank, the augmented order statistic is `+infinity` and the system fails closed to CONTINUE/HOLD. Invalid numerical state likewise cannot collapse to a narrow certificate.

On a generated-point-cloud ICP/Kabsch backend, the first frozen 160-TRAIN/160-CALIBRATION/120-TEST experiment observed 85.0% whole-trajectory coverage for a nominal 90% marginal target. Certificates occurred before estimator stopping in 90.0%, 87.5%, and 87.5% of all assigned episodes for top-suction, label-alignment, and keyed-insertion readers. Overall completion differed from estimator stopping by -2.50, -0.83, and 0.00 percentage points; zero unsafe ACTs were observed in that 120-episode numerical fixture. A raw-proxy conformal ablation attained 95.83% coverage but HOLDed every episode, demonstrating that coverage alone does not make a useful stopping certificate.

Post-review hardening adds fail-closed conformal boundary handling, finite-sample paired binary non-inferiority inference for the public lab harness, explicit separation of certificate hitting time from terminal HOLD, a matched same-split comparator experiment on new test seeds, and a backend-agnostic university protocol. Numerical trajectory-prefix costs are no longer treated as online speed measurements. Real RGB-D coverage, direct GPU/backend latency advantage, physical manipulation non-inferiority, and safety certification remain open.

---

## 1. Problem

An iterative pose backend produces

\[
(\hat P_0,r_0), (\hat P_1,r_1),\ldots,(\hat P_K,r_K),
\]

where \(\hat P_k\in SE(3)\) is the pose estimate at refinement stage \(k\), and \(r_k\) contains information available online before deciding whether to continue.

Estimator-side stopping asks

\[
C_E(\hat P_k,r_k)=STOP.
\]

The downstream question is different:

> Is every pose error still admitted by the calibrated evidence already acceptable for this task?

Let \(\widehat{\mathcal C}^{cal}_k\) be the calibrated pose-error completion set and \(\mathcal A_T\) the admissible error set for task \(T\). Define the certificate hitting time

\[
\boxed{
k_C
=
\inf\left\{k:\widehat{\mathcal C}^{cal}_k\subseteq\mathcal A_T\right\}.
}
\]

If the set is empty within the configured horizon, we take

\[
k_C=+\infty.
\]

Let \(k_E\) denote estimator-side stopping. The central event is

\[
\boxed{k_C<k_E.}
\]

This is not a claim that less perception is always better. It is a testable claim that task sufficiency and estimator convergence may occur at different stages.

The executed endpoint must be distinguished from the certificate hitting time. If a certificate occurs, the certificate policy executes at \(k_C\). If no certificate occurs, the implementation may continue to the terminal budget \(K\) and return HOLD. The terminal endpoint is not relabelled as \(k_C\).

---

## 2. Position Relative to Prior Work

Task-relevant perception is established. SPADE formalizes the joint design of perception and action under sensing cost, and task-driven manipulation systems have requested additional sensing only when current information is insufficient.

Pose-estimation error and manipulation success are also established as distinct quantities. Hietanen et al. directly benchmark pose estimation in relation to manipulation outcome; subsequent work models task-compatible pose-error regions under uncertainty.

Early stopping itself is not new. Iterative pose systems can terminate based on estimator stabilization or another estimator-side criterion. Likewise, conformal pose uncertainty is prior art: existing methods construct statistically calibrated uncertainty sets for object pose.

The present candidate contribution is narrower:

\[
\boxed{
\text{use a calibrated pose-error set as a downstream task stopping certificate}
}
\]

rather than waiting only for estimator-side convergence.

No universal priority claim is made.

---

## 3. Completion Sets and Task Equivalence

The project was motivated by a finite-readout perspective: unresolved latent states need not be identical if the declared downstream reader cannot distinguish them for the present task. Toledo/Readout materials are retained as provenance and governance, not as a proof of robotics performance.

### 3.1 Admissible completion set

**[NEW–DEFINITION]**

For retained readout \(r_k\) and declared compatibility rule \(A\), define

\[
\mathcal C_k
=
\{z:z\text{ remains compatible with }r_k\text{ under }A\}.
\]

This is a set of alternatives not yet excluded by the retained evidence. It is not a set of states asserted to be true.

### 3.2 Task-reader image

**[NEW–DERIVATION/PROPOSAL]**

\[
\mathcal Y_{T,k}
=
\{O_T(z):z\in\mathcal C_k\}.
\]

For a binary reader, a robust PASS occurs only when

\[
\mathcal Y_{T,k}=\{PASS\}.
\]

Thus latent pose identity can remain unresolved while the task verdict is resolved.

---

## 4. Calibrated Completion Envelope

### 4.1 Observable error-shape model

The numerical implementation fits six log-linear coordinate-wise error-shape models on TRAIN episodes only:

\[
\log \tilde e_{k,i}
=
\beta_i^\top x_{k,i},
\]

where \(x_{k,i}\) contains only online-observable quantities in the numerical backend: local dispersion proxy, proposed next update, residual RMS, and normalized stage. The model predicts error shape, not a calibrated posterior.

### 4.2 Whole-trajectory nonconformity

For calibration episode \(e\), define

\[
A_e
=
\max_{k\le K}\max_{i\le6}
\left[
\log(|e_{e,k,i}|+\delta_i)-\log\tilde e_{e,k,i}
\right].
\]

There is one exchangeable calibration score per complete episode. Taking the maximum over stages and coordinates makes the containment event simultaneous over the retained trajectory before the stopping stage is selected.

### 4.3 Finite split-conformal scale

Let \(A_1,\ldots,A_n\) be finite calibration scores and augment them by \(+\infty\):

\[
\mathcal S^+=\{A_1,\ldots,A_n,+\infty\}.
\]

For target miscoverage \(\alpha\), define

\[
r=\left\lceil(n+1)(1-\alpha)\right\rceil
\]

and let

\[
\boxed{
\widehat q_{1-\alpha}=\mathcal S^+_{(r)}.
}
\]

This is important at small calibration sizes. If \(r=n+1\), then

\[
\widehat q_{1-\alpha}=+\infty,
\]

so the completion envelope is uninformative and cannot license ACT. The implementation does not replace this rank by the largest finite calibration score.

### 4.4 Coordinate bounds

For finite valid model output,

\[
b_{k,i}
=
\exp(\log\tilde e_{k,i}+\widehat q_{1-\alpha})-\delta_i,
\]

and

\[
\widehat{\mathcal C}^{cal}_k
=
\{e:|e_i|\le b_{k,i}\;\forall i\}.
\]

If \(\widehat q=+\infty\), any model value is non-finite, numerical overflow occurs, or a task specification is malformed, the implementation fails closed to an unbounded certificate / no ACT. Invalid uncertainty must never be converted into zero uncertainty.

---

## 5. Coverage-to-Task Implication

### Proposition 1 — marginal whole-trajectory containment

Conditional on the standard exchangeability assumptions of split conformal prediction after the shape model is frozen,

\[
\Pr\left(
 e^{true}_k\in\widehat{\mathcal C}^{cal}_k
 \;\forall k\le K
\right)
\ge1-\alpha.
\]

The guarantee is marginal over future exchangeable episodes. It is not a deterministic or distribution-shift guarantee.

### Proposition 2 — set inclusion

If the gate returns ACT only when

\[
\widehat{\mathcal C}^{cal}_k\subseteq\mathcal A_T,
\]

then, on a covered episode,

\[
e_k^{true}\in\widehat{\mathcal C}^{cal}_k
\Rightarrow
e_k^{true}\in\mathcal A_T.
\]

This second step is set inclusion, not a probabilistic theorem.

Neither proposition establishes collision safety, controller reliability, contact success, OOD robustness, or physical robot performance unless those quantities are part of the validated reader and evidence.

---

## 6. Numerical Backend and Readers

The public numerical backend uses generated asymmetric 3-D point clouds, nearest-neighbour point-to-point ICP, Kabsch rigid alignment, and damped iterative updates.

Pose error is reported as

\[
(t_x,t_y,t_z,r_x,r_y,r_z),
\]

where \((r_x,r_y,r_z)\) are components of the **relative rotation vector** in radians. They are not globally defined Euler roll, pitch, and yaw angles. The toy task readers refer to selected local rotation-vector components.

Three readers are used:

1. **top suction** — translation plus two local rotational components; the third local rotational component is ignored by the constructed reader;
2. **label alignment** — all six relative error components must satisfy coordinate tolerances;
3. **keyed insertion** — a coupled L1-style constraint on selected translation and local rotation components:

\[
\frac{|e_x|}{\tau_x}
+
\frac{|e_y|}{\tau_y}
+
\frac{|e_{r_3}|}{\tau_{r_3}}
\le1.
\]

These are numerical admissibility readers, not physical contact models.

---

## 7. First Frozen Full Numerical Test

The first learned-shape protocol used:

```text
TRAIN        160 episodes
CALIBRATION  160 episodes
FINAL TEST   120 episodes
STRESS        60 episodes
maximum stage 20
points        120
nominal marginal whole-trajectory coverage 90%
```

Earlier development test seeds were retired before the frozen FINAL TEST.

### 7.1 Coverage

Observed whole-trajectory envelope coverage on FINAL TEST was

\[
85.00\%,
\]

with Wilson 95% interval 77.53%–90.30%. This is below the nominal 90% target and is reported without reinterpretation.

### 7.2 Certificate occurrence and completion

| Reader | certificate before estimator, all episodes | estimator completion | certificate-policy completion | HOLD | observed unsafe ACT |
|---|---:|---:|---:|---:|---:|
| top suction | 90.00% | 95.00% | 92.50% | 7.50% | 0.00% |
| label alignment | 87.50% | 92.50% | 91.67% | 8.33% | 0.00% |
| keyed insertion | 87.50% | 90.00% | 90.00% | 10.00% | 0.00% |

Zero observed unsafe ACTs do not imply zero population risk. The set-inclusion construction makes zero unsafe ACT on a *covered* numerical episode an implementation consistency condition under the matching reader; it is not an independent physical safety result.

Historical output fields that used the terminal budget endpoint for no-certificate episodes should not be interpreted as a mean certificate hitting time. Version 0.5 reports certificate hitting time only conditional on a certificate and reports terminal HOLD endpoint separately.

---

## 8. Negative Control and Shift Boundary

A raw-proxy trajectory-conformal envelope achieved 95.83% held-out coverage in its frozen numerical run but licensed no ACT for any of the three readers.

Therefore

\[
\boxed{
\text{coverage alone}\neq\text{useful stopping certificate}.
}
\]

Under a two-fold sensor-noise perturbation, the learned numerical envelope remained usable for some readers but keyed insertion returned HOLD throughout the frozen stress sample. This is an operational failure under that shift, not evidence of a general OOD detector.

---

## 9. Matched Comparator Experiment

A post-review experiment was added specifically to test whether the apparent benefit is attributable to the proposed certificate rather than to a weak estimator comparator.

All methods use the same TRAIN and CALIBRATION episodes and a new set of comparison TEST seeds not used in the first final test. Comparator parameters are frozen on CALIBRATION before TEST episodes are generated/evaluated in the script.

The compared policies are:

1. full refinement budget;
2. backend default estimator stopping;
3. fixed stage selected under a calibration-only unsafe-risk upper bound;
4. estimator/residual threshold selected under the same calibration-only risk criterion;
5. learned scalar task-score threshold using the same learned error-shape representation with calibration-only risk control;
6. raw trajectory-conformal completion envelope;
7. trajectory-conformal task certificate.

This comparison is intended to isolate the value of **set-valued task certification**, not merely better error prediction or a stricter estimator baseline. The result record is versioned separately in `results/matched_baselines/` or CI artifacts. It remains generated-data evidence.

---

## 10. Repeated-Cycle Diagnostic

A ten-cycle numerical diagnostic reruns TRAIN→CALIBRATION→TEST without method tuning, but it uses a reduced configuration:

```text
TRAIN 80
CALIBRATION 80
TEST 60
maximum stage 16
points 80
```

This differs from the first 160/160/120, K=20, 120-point configuration. It is therefore labelled

```text
[NumericalReplication-ReducedConfiguration]
```

and must not be used to reinterpret the first final-test coverage realization. Its role is mechanical stress-testing of the procedure under a smaller declared numerical setting.

---

## 11. Timing: Prefix Cost Is Not Online Speedup

The numerical experiments generate a complete refinement trajectory and sum recorded stage costs only to a counterfactual stopping endpoint. This provides a trajectory-prefix cost estimate.

It does not execute two independently stopped online policies and therefore does not establish a latency advantage.

For a real speed claim, the public university harness requires a separate mode in which certificate-stop and estimator-stop loops are executed and timed directly under a paired design. The measurement must include feature extraction, refinement updates, certificate/gate overhead, synchronization, and other policy-specific work.

Only direct online-policy timing can support

\[
T_C<T_E.
\]

---

## 12. Completion Non-Inferiority for Real-System Tests

The public lab harness requires a predeclared non-inferiority margin \(\Delta\), confidence level, minimum sample size, and independent sampling unit.

For paired binary completion outcomes, define

\[
p_{10}=P(C=1,E=0),\qquad p_{01}=P(C=0,E=1),
\]

so that the paired completion difference is

\[
\delta=p_{10}-p_{01}.
\]

The reference implementation constructs a conservative finite-sample lower confidence bound for \(\delta\) from exact Clopper–Pearson marginal bounds on the paired discordances with Bonferroni allocation. Non-inferiority is declared only when

\[
LCB(\delta)\ge-\Delta
\]

and the predeclared minimum sample size has been reached.

A one-episode equality or a degenerate zero-discordance bootstrap is therefore insufficient.

---

## 13. University-Lab Falsification Protocol

Any iterative 6D backend can be tested if it exposes intermediate online features, its estimator-side stopping decision, and an independent pose reference for offline calibration/evaluation.

The lab exports three disjoint files:

```text
train.jsonl
calibration.jsonl
test.jsonl
```

and runs

```bash
python lab/run_real_system.py \
  --config lab/config.example.json \
  --train train.jsonl \
  --calibration calibration.jsonl \
  --test test.jsonl \
  --out lab_results.json
```

The standard report contains coverage, certificate rate, certificate-before-estimator rate, certificate hitting time conditional on a hit, terminal HOLD endpoint, paired completion difference, finite-sample non-inferiority bound, HOLD, unsafe ACT, prefix cost, and measured online-policy latency only when supplied.

A real sensor/backend result does not imply physical manipulation success. Physical manipulation non-inferiority requires actual robot execution with an outcome measured independently of the pose gate.

---

## 14. Limitations

The current research evidence is numerical. No camera, RGB-D sensor, neural pose estimator, contact dynamics, or physical robot is used in the reported numerical experiments.

The learned shape model is intentionally simple and may fail on real pose-refinement trajectories. The conformal interpretation depends on the declared exchangeability setting. Distribution shifts can degrade coverage or make certificates too conservative.

The numerical readers are constructed error-space criteria, not validated physical task-success models. Set inclusion is meaningful only relative to the reader actually declared.

The matched comparison strengthens attribution inside the numerical fixture but cannot establish superiority on real perception systems. The current work should therefore be treated as a reproducible method and falsification protocol awaiting external real-backend tests.

---

## 15. Conclusion

The paper isolates one testable distinction:

\[
\boxed{
\text{estimator convergence}\neq\text{task sufficiency}.
}
\]

The proposed stopping rule does not require the pose to be known exactly. It requires the calibrated set of pose errors still compatible with the retained evidence to lie entirely inside the downstream task-admissible set.

The post-review implementation now enforces that claim conservatively at critical boundary cases: insufficient finite calibration yields an unbounded envelope, invalid numerics fail closed, no-certificate episodes remain distinct from certificate hitting times, and population non-inferiority cannot be declared from a degenerate tiny sample. Numerical prefix timing is no longer promoted into a speed claim.

The next decisive evidence is external and simple to state: instrument a real iterative 6D backend, calibrate on real trajectories, lock a final test, and measure whether

\[
k_C<k_E,
\]

while task-admissibility completion satisfies the predeclared non-inferiority criterion and direct online-policy timing demonstrates a real latency reduction.

If those conditions fail, the proposal fails for that backend/task operating point. If they hold across independent systems, the numerical mechanism becomes an empirical robotics result rather than only a public numerical prototype.

---

## References

Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*.

Hibbard, M., Tanaka, T., & Topcu, U. (2023). Simultaneous perception–action design via invariant finite belief sets. *Automatica, 155*, 111140.

Hietanen, A., Latokartano, J., Foi, A., Pieters, R., Kyrki, V., Lanz, M., & Kämäräinen, J.-K. (2021). Benchmarking pose estimation for robot manipulation. *Robotics and Autonomous Systems, 143*, 103810.

Mitash, C., Shome, R., Wen, B., Boularias, A., & Bekris, K. (2020). Task-driven perception and manipulation for constrained placement of unknown objects. *IEEE Robotics and Automation Letters, 5*(4), 5605–5612.

Naik, L., Iversen, T. M., Kramberger, A., & Krüger, N. (2025). Robotic task success evaluation under multi-modal non-parametric object pose uncertainty. *Industrial Robot, 52*(5), 651–662.

Wen, B., Yang, W., Kautz, J., & Birchfield, S. (2024). FoundationPose: Unified 6D pose estimation and tracking of novel objects. *CVPR 2024*.
