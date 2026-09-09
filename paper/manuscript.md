# Coverage-Qualified Task Stopping for Iterative 6D Pose Refinement

**Yaoharee Lahtee**  
Open Civil Science Initiative, Bangkok, Thailand

Version 0.6 — reproducible numerical study and real-system test protocol, 9 September 2026

## Abstract

Iterative 6D pose pipelines usually stop at a fixed refinement budget or an estimator-side convergence rule. A downstream task can become insensitive to the unresolved pose alternatives earlier. We study one question: **can refinement stop when a calibrated set of pose errors is already entirely admissible for the declared downstream task, even though the estimator would continue?**

We define a certificate hitting time \(k_C\) as the first refinement stage whose whole calibrated pose-error completion set lies inside a task-admissible set, and compare it with estimator stopping \(k_E\). The certificate is constructed by fitting an observable error-shape model on TRAIN trajectories and conformalizing one maximum nonconformity score per complete CALIBRATION trajectory. The online gate sees no hidden ground truth. Finite-sample boundary cases fail closed: if the split-conformal rank requires the augmented \(+\infty\) score, or certificate numerics are invalid, the system cannot ACT.

On a generated-point-cloud ICP/Kabsch backend, the first frozen 160-TRAIN/160-CALIBRATION/120-TEST study observed 85.0% whole-trajectory coverage for a nominal 90% marginal target. Certificates occurred before estimator stopping in 90.0%, 87.5%, and 87.5% of all episodes for top-suction, label-alignment, and keyed-insertion readers. Overall completion differed from estimator stopping by -2.50, -0.83, and 0.00 percentage points, respectively.

A post-review matched comparison on 120 new test episodes provides a more discriminating result. A learned scalar threshold stopped earlier than the set-valued certificate on all three readers, but produced more unsafe ACTs. For keyed insertion, the trajectory-conformal certificate matched estimator-default completion at 95%, changed the observed 5% unsafe-ACT rate into 5% HOLD, and used a mean executed endpoint of 11.025 stages versus 13.100 for estimator stopping. For top suction and label alignment, the certificate likewise traded a small amount of abstention for lower observed unsafe ACT than the scalar comparator. Thus the numerical evidence supports a **coverage/risk-qualified stopping trade-off**, not universal earliest stopping or superiority.

The repository includes a backend-agnostic university harness with paired binary non-inferiority inference and a separate mode for direct online-policy timing. Numerical trajectory-prefix cost is not treated as a speed measurement. Real RGB-D coverage, direct neural-backend/GPU latency advantage, physical manipulation non-inferiority, and safety certification remain open.

---

## 1. Problem

An iterative pose backend produces

\[
(\hat P_0,r_0),(\hat P_1,r_1),\ldots,(\hat P_K,r_K),
\]

where \(\hat P_k\in SE(3)\) is the stage-\(k\) estimate and \(r_k\) contains only information available before deciding whether to continue.

Estimator stopping asks whether the estimator itself is finished. The downstream task asks whether any pose error still compatible with the evidence can change the task verdict.

Let \(\widehat{\mathcal C}^{cal}_k\) be a calibrated pose-error completion set and \(\mathcal A_T\) the declared admissible error set for task \(T\). Define

\[
\boxed{
k_C=\inf\{k:\widehat{\mathcal C}^{cal}_k\subseteq\mathcal A_T\}.
}
\]

If no certificate occurs within the refinement budget, \(k_C=+\infty\). Let \(k_E\) be estimator-side stopping. The computational event of interest is

\[
\boxed{k_C<k_E.}
\]

The executed endpoint is a different variable: when \(k_C<\infty\), the certificate policy ACTs at \(k_C\); otherwise it may continue to budget \(K\) and return HOLD. A terminal HOLD endpoint is never relabelled as \(k_C\).

---

## 2. Position Relative to Prior Work

The broad ingredients are established. Task-relevant sensing and perception cost are prior work; pose-estimation error and manipulation success are known to be distinct; iterative pose methods already use estimator-side early stopping; and conformal methods have been applied to pose uncertainty.

The candidate contribution is therefore narrow:

\[
\boxed{
\text{use a calibrated pose-error set as a downstream stopping certificate}
}
\]

rather than stopping only when the estimator converges or when a scalar score crosses a threshold.

This paper does not claim to invent task-aware perception, early stopping, conformal prediction, or pose uncertainty sets.

---

## 3. Method

### 3.1 Observable error-shape model

The numerical reference implementation learns six coordinate-wise log-error shape models from TRAIN episodes only:

\[
\log \tilde e_{k,i}=\beta_i^\top x_{k,i},
\]

where \(x_{k,i}\) contains online-observable refinement features. In the reference backend these are a local dispersion proxy, proposed next update, residual RMS, and normalized stage.

The learned shape is not treated as calibrated uncertainty.

### 3.2 Whole-trajectory calibration score

For calibration episode \(e\), define one score

\[
A_e
=
\max_{k\le K}\max_{i\le6}
\left[
\log(|e_{e,k,i}|+\delta_i)-\log\tilde e_{e,k,i}
\right].
\]

The maximum over all retained stages is intentional: the eventual stopping stage is selected adaptively, so calibration is performed at the episode/trajectory level rather than treating stages as independent calibration samples.

### 3.3 Finite split-conformal scale

Augment the \(n\) finite calibration scores by \(+\infty\):

\[
\mathcal S^+=\{A_1,\ldots,A_n,+\infty\}.
\]

For target miscoverage \(\alpha\), let

\[
r=\lceil(n+1)(1-\alpha)\rceil,
\qquad
\widehat q=\mathcal S^+_{(r)}.
\]

If \(r=n+1\), then \(\widehat q=+\infty\), so the completion set is uninformative and cannot license ACT. The implementation never caps the rank at \(n\).

### 3.4 Completion envelope

For finite valid quantities,

\[
b_{k,i}
=
\exp(\log\tilde e_{k,i}+\widehat q)-\delta_i,
\]

and

\[
\widehat{\mathcal C}^{cal}_k
=
\{e:|e_i|\le b_{k,i}\;\forall i\}.
\]

NaN, non-finite model state, malformed task tolerances, or overflow fail closed to an unbounded envelope/no ACT. Invalid certificate state must never become zero-width certainty.

### 3.5 Task certificate

The online gate returns ACT only if

\[
\widehat{\mathcal C}^{cal}_k\subseteq\mathcal A_T.
\]

Otherwise it continues refinement; if no certificate is reached within budget it returns HOLD.

Under the standard exchangeability assumptions of split conformal prediction, the whole-trajectory envelope has marginal coverage at the declared level. Conditional on the true pose error being inside the retained envelope, the implication from robust set inclusion to the declared numerical task reader is deterministic. Neither step establishes physical robot safety unless the reader and evidence actually encode that outcome.

---

## 4. Numerical Backend and Readers

The public numerical backend uses generated asymmetric 3-D point clouds, nearest-neighbour point-to-point ICP, Kabsch alignment, and damped iterative updates.

Pose error is represented as

\[
(t_x,t_y,t_z,r_x,r_y,r_z),
\]

where \((r_x,r_y,r_z)\) are components of a **relative rotation vector** in radians, not globally defined Euler roll/pitch/yaw angles.

Three numerical readers are used:

- **top suction:** translation plus two local rotational components; the third local rotational component is irrelevant to the constructed reader;
- **label alignment:** all six relative error components satisfy declared tolerances;
- **keyed insertion:** a coupled L1-style constraint on two translation components and one local rotational component.

These are numerical error-space readers, not physical contact models.

---

## 5. First Frozen Full Numerical Test

The first learned certificate used:

```text
TRAIN        160
CALIBRATION  160
FINAL TEST   120
STRESS        60
maximum stage 20
points        120
nominal whole-trajectory marginal coverage 90%
```

Observed FINAL TEST whole-trajectory coverage was 85.00% (Wilson 95% CI 77.53%–90.30%), below the nominal target.

| Reader | cert < estimator, all episodes | certificate rate | certificate k, mean given hit | estimator completion | certificate completion | HOLD | unsafe ACT |
|---|---:|---:|---:|---:|---:|---:|---:|
| top suction | 90.00% | 92.50% | 9.568 | 95.00% | 92.50% | 7.50% | 0.00% |
| label alignment | 87.50% | 91.67% | 10.073 | 92.50% | 91.67% | 8.33% | 0.00% |
| keyed insertion | 87.50% | 90.00% | 10.833 | 90.00% | 90.00% | 10.00% | 0.00% |

Zero observed unsafe ACTs do not imply zero population risk. They also do not validate physical task success: the task reader is part of the numerical fixture.

---

## 6. Negative Control

A simpler raw-proxy split-conformal envelope achieved 95.83% held-out whole-trajectory coverage but licensed no ACT for any reader.

\[
\boxed{
\text{coverage alone}\neq\text{operationally useful certificate}.
}
\]

The retained set must be sufficiently covered **and** sufficiently narrow in task-relevant directions.

---

## 7. Matched Comparator Study

A post-review experiment was added to isolate why the method appears useful. It uses the same TRAIN=160 and CALIBRATION=160 episodes for all methods and 120 new comparison TEST episodes from seeds not used in the first final test. Comparator parameters are selected on CALIBRATION before TEST is generated/evaluated in the script.

Compared policies are:

1. full budget;
2. backend default estimator stop;
3. calibration-risk-checked fixed stage;
4. calibration-risk-checked estimator threshold;
5. calibration-risk-checked learned scalar task score using the same learned shape representation;
6. raw calibrated completion envelope;
7. trajectory-conformal completion certificate.

### 7.1 Top suction

| Policy | mean endpoint k | completion | HOLD | unsafe ACT |
|---|---:|---:|---:|---:|
| estimator default | 13.100 | 97.50% | 0.00% | 2.50% |
| estimator threshold | 11.300 | 95.00% | 0.00% | 5.00% |
| learned scalar | **8.283** | 95.00% | 2.50% | 2.50% |
| **trajectory-conformal certificate** | 9.708 | **96.67%** | 3.33% | **0.00%** |

The scalar policy stops earlier, but the set-valued certificate is more conservative and has higher observed completion with fewer unsafe ACTs in this numerical test.

### 7.2 Label alignment

| Policy | mean endpoint k | completion | HOLD | unsafe ACT |
|---|---:|---:|---:|---:|
| estimator default | 13.100 | 95.83% | 0.00% | 4.17% |
| learned scalar | **9.025** | 94.17% | 2.50% | 3.33% |
| **trajectory-conformal certificate** | 10.167 | **95.83%** | 3.33% | **0.83%** |

Again, the scalar policy stops earlier; the certificate trades additional abstention for lower observed unsafe ACT while matching estimator-default completion in this sample.

### 7.3 Keyed insertion

| Policy | mean endpoint k | completion | HOLD | unsafe ACT |
|---|---:|---:|---:|---:|
| estimator default | 13.100 | 95.00% | 0.00% | 5.00% |
| learned scalar | **9.800** | 92.50% | 3.33% | 4.17% |
| **trajectory-conformal certificate** | 11.025 | **95.00%** | 5.00% | **0.00%** |

For keyed insertion, the certificate matches estimator-default completion while converting the observed estimator failures into HOLD and using a lower mean executed endpoint.

The matched study therefore rejects a stronger but unsupported interpretation that the certificate is the fastest stopping policy. The defensible numerical finding is a **coverage/risk-qualified trade-off**: the set-valued certificate is more conservative than a scalar score and, in these matched tests, generally reduces unsafe ACT at the cost of some HOLD and a later endpoint than the scalar comparator.

Authoritative frozen record: `results/matched_baselines/STANDARD_RESULTS.json` and `STANDARD_RESULTS.md`.

---

## 8. Timing and Statistical Claims

### 8.1 Prefix cost is not speed

The numerical experiments generate complete trajectories and sum recorded stage costs to counterfactual policy endpoints. These are trajectory-prefix cost estimates.

They do not establish an online speedup because they do not execute two independently stopped policy loops with all policy-specific overhead.

A speed claim requires direct paired execution timing including feature extraction, update cost, certificate/gate cost, synchronization, and relevant device timing.

### 8.2 Non-inferiority in the public lab harness

For paired binary task outcomes, define

\[
p_{10}=P(C=1,E=0),\qquad p_{01}=P(C=0,E=1),
\]

with paired difference \(\delta=p_{10}-p_{01}\).

The backend-agnostic evaluator uses a conservative lower confidence bound built from Clopper–Pearson bounds on the paired discordances with Bonferroni allocation. It requires a predeclared non-inferiority margin, confidence level, independent sampling unit, and minimum FINAL TEST size. A one-episode equality cannot certify population non-inferiority.

---

## 9. Reduced Repeated-Cycle Diagnostic

Ten additional cycles used TRAIN=80, CALIBRATION=80, TEST=60, maximum stage 16, and 80 points. Mean realized coverage was 91.17%, but this configuration differs from the first full numerical test.

The record is therefore labelled

```text
[NumericalReplication-ReducedConfiguration]
```

and is used only as a repeatability stress-test for that reduced setting. It is not evidence that the first 85% final-test realization should be replaced or reinterpreted.

---

## 10. Real-System Falsification Protocol

A university lab can test any iterative 6D backend that exposes intermediate online features and estimator-side stopping, provided an independent pose reference is available for offline TRAIN/CALIBRATION/FINAL TEST evaluation.

The public evaluator consumes:

```text
train.jsonl
calibration.jsonl
test.jsonl
```

and reports:

```text
held-out whole-trajectory coverage
certificate rate
k_C < k_E
certificate hitting time conditional on hit
terminal HOLD endpoint
paired completion difference
finite-sample non-inferiority bound
HOLD
unsafe ACT
trajectory-prefix cost
measured online-policy latency only when supplied
```

Default timing mode deliberately disables a latency PASS. Direct latency inference is enabled only when separately executed paired policy timings are supplied.

A real sensor/backend result does not imply physical manipulation success. Physical manipulation claims require actual robot executions with outcomes measured independently of the pose gate.

---

## 11. Limitations

The current research evidence is numerical. No camera, RGB-D sensor, iterative neural pose model, contact dynamics, or physical robot is used in the reported research experiments.

The learned error-shape model is intentionally simple. Exchangeability can fail. The box/L1 readers are constructed numerical outcome definitions rather than validated physical success models. Matched numerical comparisons improve causal attribution inside the fixture but cannot establish superiority on real systems.

Observed zero unsafe ACTs in a finite sample do not establish zero risk. Conformal marginal coverage is not deterministic safety certification. Distribution-shift behavior requires separate testing and potentially recalibration.

The present result should therefore be treated as a reproducible method, a set of negative and matched controls, and a standardized falsification protocol awaiting external real-backend experiments.

---

## 12. Conclusion

The central distinction is

\[
\boxed{
\text{estimator convergence}\neq\text{task sufficiency}.
}
\]

Coverage-qualified stopping makes the second quantity operational: stop only when the calibrated set of pose errors still admitted by the evidence lies wholly inside the declared task-admissible set.

The post-review evidence narrows rather than inflates the claim. A scalar learned score can stop earlier than the proposed set-valued certificate. The certificate's numerical value is instead its more conservative risk behavior: in the matched fixture it generally exchanges some abstention and a later endpoint than the scalar comparator for fewer unsafe ACTs, while still ending earlier than estimator stopping in useful cases.

The decisive next test is external: connect a real iterative 6D backend, freeze TRAIN/CALIBRATION/FINAL TEST roles, measure whether certificates occur before estimator stopping, test task completion against a predeclared non-inferiority criterion, and directly time the independently executed policies. If those conditions do not hold, the method fails for that operating point. If they hold across independent systems, the contribution moves from a numerical methods result to an empirical robotics result.

---

## References

Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*.

Hibbard, M., Tanaka, T., & Topcu, U. (2023). Simultaneous perception–action design via invariant finite belief sets. *Automatica, 155*, 111140.

Hietanen, A., Latokartano, J., Foi, A., Pieters, R., Kyrki, V., Lanz, M., & Kämäräinen, J.-K. (2021). Benchmarking pose estimation for robot manipulation. *Robotics and Autonomous Systems, 143*, 103810.

Mitash, C., Shome, R., Wen, B., Boularias, A., & Bekris, K. (2020). Task-driven perception and manipulation for constrained placement of unknown objects. *IEEE Robotics and Automation Letters, 5*(4), 5605–5612.

Naik, L., Iversen, T. M., Kramberger, A., & Krüger, N. (2025). Robotic task success evaluation under multi-modal non-parametric object pose uncertainty. *Industrial Robot, 52*(5), 651–662.

Wen, B., Yang, W., Kautz, J., & Birchfield, S. (2024). FoundationPose: Unified 6D pose estimation and tracking of novel objects. *CVPR 2024*.
