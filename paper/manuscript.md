# Coverage-Qualified Task Stopping for Iterative 6D Pose Refinement under Incomplete State Knowledge

**Yaoharee Lahtee**  
Open Civil Science Initiative, Bangkok, Thailand

Version 0.4 — executable numerical manuscript, 9 September 2026

## Abstract

Iterative 6D object-pose estimation is normally terminated by a fixed refinement budget or an estimator-side convergence rule. A downstream manipulation task can become executable before either condition is reached. This paper studies a narrower stopping problem: whether refinement can terminate when **every pose state still admitted by a calibrated completion envelope yields the same downstream task verdict**, even though the estimator itself would continue refining.

The proposal combines two established ideas without claiming either as new: task-relevant perception and calibrated pose uncertainty. Prior work already shows that sensing cost can be conditioned on task relevance, that pose accuracy and manipulation success are distinct, and that conformal methods can produce statistically calibrated object-pose uncertainty sets. Early stopping inside iterative pose pipelines is also prior art. The specific system studied here instead treats a calibrated pose set as a **downstream stopping certificate**.

We implement the method on an open numerical 6D backend using iterative point-to-point ICP/Kabsch registration on generated asymmetric 3-D point clouds. A learned observable error-shape model is fitted on 160 TRAIN episodes, a trajectory-level split-conformal scale is frozen on 160 disjoint CALIBRATION episodes, and the resulting gate is evaluated on 120 previously unused FINAL TEST episodes. The gate never receives hidden ground-truth pose. At a nominal 90% marginal whole-trajectory coverage target, the first frozen final test observes 85.0% trajectory-envelope coverage (Wilson 95% CI 77.53–90.30%). Despite that imperfect empirical coverage, the completion certificate stops before estimator convergence in 87.5–90.0% of episodes across top-suction, label-alignment, and keyed-insertion readers. Mean stopping stage decreases from 13.50 to 10.35, 10.90, and 11.75, respectively. Overall task completion differs from estimator-side stopping by -2.50, -0.83, and 0 percentage points, with zero unsafe ACT decisions in the 120-episode numerical test.

A deliberately simpler raw-proxy conformal envelope provides an important negative control: it attains 95.83% held-out trajectory coverage yet returns HOLD for every episode and every task. Thus coverage alone is insufficient; a useful completion set must be both sufficiently covered and sufficiently task-discriminative. Under a two-fold sensor-noise shift, the learned envelope remains usable for suction and label alignment but returns HOLD for all keyed-insertion episodes, exposing a task-dependent distribution-shift boundary.

The present evidence supports a public numerical mechanism, not a physical-robot claim. Real RGB-D coverage, GPU latency reduction, physical manipulation non-inferiority, and safety certification remain open hypotheses.

---

## 1. Problem

An iterative pose estimator produces a sequence

\[
(\hat P_0,r_0), (\hat P_1,r_1),\ldots,(\hat P_K,r_K),
\]

where \(\hat P_k\in SE(3)\) is the current pose estimate and \(r_k\) is the observable refinement readout available at stage \(k\).

A conventional stopping rule asks whether the estimator itself is finished:

\[
C_E(\hat P_k,r_k)=STOP.
\]

A manipulation controller asks a different question:

> Is every pose still compatible with the current evidence already acceptable for this task?

These conditions need not become true at the same refinement stage.

For example, unresolved yaw can remain irrelevant to a rotationally symmetric top-suction task while remaining decisive for label alignment. Likewise, small independent coordinate errors can jointly violate an insertion constraint.

We therefore distinguish estimator-side stopping from **coverage-qualified task stopping**.

Let \(k_E\) denote the estimator stopping stage. Let \(\widehat{\mathcal C}^{cal}_k\) denote a calibrated completion set for the pose error at stage \(k\), and let \(O_T\) denote the downstream task reader. Define

\[
\boxed{
k_C
=
\min\left\{k:
O_T(e)=PASS\;\;\forall e\in\widehat{\mathcal C}^{cal}_k
\right\}.
}
\]

The computational opportunity is

\[
\boxed{k_C<k_E.}
\]

The central hypothesis is not that less perception is always better. It is that a task may become invariant to the unresolved pose alternatives before a generic pose estimator reaches its own stopping condition.

---

## 2. Position Relative to Prior Work

### 2.1 Task-relevant perception is established

Task-dependent sensing is not new. Mitash et al. couple perception and manipulation for constrained placement, requesting additional sensing when current geometric information is insufficient and explicitly seeking to reduce sensing and execution time. Hibbard, Tanaka, and Topcu formulate Simultaneous Perception–Action Design (SPADE), in which sensing itself is optimized against task cost and only task-relevant information need be transmitted to the downstream controller.

The present work therefore does not claim the principle “perceive only what the task needs.”

### 2.2 Pose error and task success are established as distinct quantities

Hietanen et al. show that standard geometric pose metrics do not directly determine manipulation success and introduce task-linked evaluation. Naik et al. subsequently model both pose uncertainty and task-acceptable error space. These results rule out treating task-dependent pose admissibility as a new concept.

### 2.3 Early stopping is established

Iterative pose methods already terminate computation early for estimator-side reasons. AnyBox, for example, uses an early-stopping mechanism inside an iterative pose-and-scale estimation pipeline for warehouse boxes. Thus “pose refinement can stop early” is not the contribution here.

### 2.4 Calibrated 6D pose uncertainty is established

Yang and Pavone apply inductive conformal prediction to keypoint detections and propagate the resulting constraints into a Pose UnceRtainty SEt with statistical coverage guarantees. Wang et al. later use inductive conformal prediction and deterministic geometric propagation to obtain compact 6D pose confidence regions. These works make clear that conformal pose uncertainty itself is prior art.

The present question lies at the intersection of these lines:

\[
\boxed{
\text{Can a calibrated pose set become a downstream stopping certificate?}
}
\]

The proposed composition is specific: iterative pose refinement is terminated not because the estimate has converged and not merely because a scalar confidence score is small, but because the entire calibrated set of still-admissible pose errors lies within one task-reader equivalence class.

No universal priority claim is made.

---

## 3. Completion Sets and Task Equivalence

### 3.1 Toledo anchors and equation status

This manuscript uses Toledo only as a governance and finite-readout anchor. Existing Toledo equations are not silently modified. Robotics-specific constructions below are labeled as new project definitions or derivations.

**[EXISTING–TOLEDO: weld/M.01.v1]** retained-state transition:

\[
S_{n+1}=F(S_n,u_n,c_n,T_n).
\]

**[EXISTING–TOLEDO: weld/M.02.v1]** domain weld and reader preservation:

\[
q_D(F(z,u,c,T))=F_D^{\#}(q_D(z),u,c,T),
\]

\[
O_D(z;Q,c)=O_D^{\#}(q_D(z);Q,c).
\]

**[EXISTING–TOLEDO: weld/M.03.v1]** finite-horizon reader equivalence:

\[
z\sim_{Q,O,c,L}z'
\iff
O(F^kz)=O(F^kz')\quad\forall k\le L.
\]

These anchors motivate, but do not prove, the robotics construction.

### 3.2 Admissible completion set

**[NEW–DEFINITION]**

For retained readout \(r_k\), context \(c\), and declared admissibility rule \(A\), define

\[
\mathcal C_k
=
\{z:\;z\text{ remains compatible with }r_k\text{ under }A\}.
\]

This is not a set of states asserted to be true. It is a computational representation of latent alternatives not yet excluded by the current record.

### 3.3 Task-reader image

**[NEW–DERIVATION/PROPOSAL]**

\[
\mathcal Y_{T,k}
=
\{O_T(z):z\in\mathcal C_k\}.
\]

For a binary task reader,

\[
\mathcal Y_{T,k}\subseteq\{PASS,FAIL\}.
\]

The robust verdict is

\[
V_{T,k}=
\begin{cases}
PASS,&\mathcal Y_{T,k}=\{PASS\},\\
FAIL,&\mathcal Y_{T,k}=\{FAIL\},\\
HOLD,&\mathcal Y_{T,k}=\{PASS,FAIL\}.
\end{cases}
\]

A task may therefore be resolved while latent pose identity remains unresolved.

### 3.4 Task ambiguity

For a metric \(d_T\) on task readouts, define

\[
W_T(\mathcal C_k)
=
\sup_{z,z'\in\mathcal C_k}
 d_T(O_T(z),O_T(z')).
\]

For a binary reader, \(W_T=0\) means the remaining alternatives no longer change the task verdict. It does not mean that the pose is known exactly.

---

## 4. From Observable Refinement Readout to a Calibrated Completion Envelope

### 4.1 Why the raw proxy is insufficient

The numerical ICP backend exposes a local linearized dispersion proxy. That quantity is observable but is neither a posterior distribution nor a calibrated confidence region.

A first conformal ablation directly scaled this proxy with one trajectory-level conformal factor. It achieved 95.83% held-out trajectory coverage but produced envelopes so wide that all three task readers returned HOLD on every episode.

This failure is scientifically useful:

\[
\boxed{
\text{coverage alone}\neq\text{operationally useful task certificate}.
}
\]

### 4.2 Frozen observable shape model

The stronger method separates TRAIN, CALIBRATION, and FINAL TEST.

On TRAIN episodes only, six log-linear models estimate the *shape* of coordinate-wise absolute pose error from online-observable quantities:

\[
\log \tilde e_{k,i}
=
\beta_i^\top
\begin{bmatrix}
1\\
\log(p_{k,i}+\epsilon^p_i)\\
\log(|\Delta_{k,i}|+\epsilon^\Delta_i)\\
\log(RMS_k+\epsilon^r)\\
k/K
\end{bmatrix},
\]

where \(p_{k,i}\) is the local proxy and \(\Delta_{k,i}\) is the proposed next refinement update.

The model is frozen before conformal calibration.

### 4.3 One score per complete trajectory

**[NEW–DEFINITION; external statistical basis: split conformal prediction]**

For calibration episode \(e\), define

\[
A_e
=
\max_{k\le K}\max_{i\le6}
\left[
\log(|e_{e,k,i}|+\delta_i)
-
\log \tilde e_{e,k,i}
\right].
\]

The maximum is taken over every retained stage and all six pose coordinates.

This choice is deliberate. The stopping stage is selected adaptively later; treating each stage as an independent calibration sample would ignore that selection. By calibrating one maximum score per whole trajectory, the containment event is defined simultaneously across all stages before any stopping stage is chosen.

### 4.4 Split-conformal scale

For \(n\) calibration episodes and target miscoverage \(\alpha\), sort

\[
A_{(1)}\le\cdots\le A_{(n)}
\]

and freeze

\[
\widehat q_{1-\alpha}=A_{(r)},
\qquad
r=\min\{n,\lceil(n+1)(1-\alpha)\rceil\}.
\]

The stage-wise coordinate bounds become

\[
\boxed{
b_{k,i}
=
\exp(\log\tilde e_{k,i}+\widehat q_{1-\alpha})-\delta_i.}
\]

The calibrated completion box is

\[
\boxed{
\widehat{\mathcal C}^{cal}_k
=
\{e:\ |e_i|\le b_{k,i}\;\forall i\}.
}
\]

The online gate receives \(r_k\), the frozen shape model, the frozen conformal scale, and the task. It does not receive hidden ground-truth pose.

---

## 5. Conditional Coverage-to-Task Certificate

### Proposition 1 — whole-trajectory marginal containment

**[NEW–DERIVATION/PROPOSAL; conditional on standard split-conformal exchangeability assumptions]**

If a future episode is exchangeable with the calibration episodes after the shape model has been frozen, standard split conformal coverage of the scalar episode score implies

\[
\Pr\left(
 e_k^{true}\in\widehat{\mathcal C}^{cal}_k
 \quad\forall k\le K
\right)
\ge 1-\alpha.
\]

The reason is algebraic: the event \(A_{new}\le\widehat q\) is exactly the event that every hidden coordinate at every retained stage lies inside the corresponding completion bound.

This is marginal, not conditional, coverage. It is not a deterministic safety guarantee.

### Proposition 2 — covered robust ACT implies numerical task PASS

If the online gate returns ACT only when

\[
O_T(e)=PASS
\quad\forall e\in\widehat{\mathcal C}^{cal}_k,
\]

then on an episode whose true hidden pose error is contained in the completion envelope,

\[
ACT_k
\Rightarrow
O_T(e_k^{true})=PASS.
\]

This is set inclusion rather than a probabilistic argument.

Combining the two propositions yields the conditional numerical implication

\[
\Pr(\text{unsafe ACT caused by violation of the declared pose reader})
\le\alpha,
\]

provided the conformal assumptions hold and the task reader itself is correct for the outcome being claimed.

The statement does **not** bound collision risk, controller failure, perception-domain shift, contact failure, or physical robot risk.

---

## 6. Numerical Backend and Tasks

The public backend performs point-to-point ICP using nearest-neighbour correspondences and Kabsch rigid alignment on generated asymmetric 3-D point clouds. A damped update produces an iterative sequence of 6D estimates.

Three numerical downstream readers are used.

### Top suction

Task-relevant variables are translation plus roll and pitch. Yaw is irrelevant within this toy reader.

### Label alignment

All six local pose-error coordinates must lie within declared coordinate tolerances.

### Keyed insertion

A coupled L1-style task reader is used on \(x\), \(y\), and yaw:

\[
\frac{|x|}{\tau_x}
+
\frac{|y|}{\tau_y}
+
\frac{|yaw|}{\tau_{yaw}}
\le1.
\]

This prevents independent marginal thresholds from silently standing in for joint task feasibility.

---

## 7. Experimental Protocol

The first frozen learned-shape final test uses:

- 160 TRAIN episodes;
- 160 disjoint CALIBRATION episodes;
- nominal marginal whole-trajectory coverage target \(1-\alpha=0.90\);
- 120 FINAL TEST episodes across three previously unused seeds;
- 60 additional episodes under a declared 2x sensor-noise shift;
- 20 maximum refinement stages;
- 120 generated points per episode.

Earlier development test seeds had already been inspected while designing the learned shape and were explicitly retired. They are recorded in the public lineage and are not reused as FINAL TEST.

The frozen learned method is compared primarily against estimator-side stopping on the same generated refinement trajectories. The raw-proxy conformal construction is retained as an ablation.

The primary readouts are:

- whole-trajectory envelope coverage;
- \(k_C\) and \(k_E\);
- rate of \(k_C<k_E\);
- overall completion with HOLD in the denominator;
- HOLD rate;
- unsafe ACT rate;
- success conditional on ACT;
- complete numerical perception-to-stop timing.

Timing is machine-specific and is secondary to iteration and outcome readouts.

---

## 8. Results

### 8.1 Raw-proxy calibrated envelope: coverage without usefulness

The raw-proxy conformal ablation attains 95.83% whole-trajectory coverage on its frozen in-distribution evaluation, but its completion sets remain too wide to license ACT for any of the three tasks.

Thus:

\[
ACT\ rate=0,\qquad HOLD\ rate=1.
\]

This rejects the idea that a high-coverage set is automatically a useful stopping certificate.

### 8.2 Learned-shape final test

The learned shape is trained before calibration. With \(\alpha=0.10\), 160 calibration episodes yield

\[
\widehat q=1.9851453152702532.
\]

On 120 previously unused in-distribution episodes, observed whole-trajectory containment is

\[
102/120=85.00\%,
\]

with Wilson 95% interval

\[
77.53\%\text{ to }90.30\%.
\]

This observed proportion is below the nominal 90% marginal target. The interval includes 90%, but the run is not reported as “achieving 90% empirical coverage.”

| Task | Estimator mean \(k_E\) | Completion mean \(k_C\) | \(k_C<k_E\) | Estimator completion | Completion-stop completion | HOLD | Unsafe ACT |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top suction | 13.500 | 10.350 | 90.00% | 95.00% | 92.50% | 7.50% | 0.00% |
| Label alignment | 13.500 | 10.900 | 87.50% | 92.50% | 91.67% | 8.33% | 0.00% |
| Keyed insertion | 13.500 | 11.750 | 87.50% | 90.00% | 90.00% | 10.00% | 0.00% |

The stopping-stage reductions are therefore

\[
\Delta k=-3.15,-2.60,-1.75
\]

for suction, alignment, and insertion.

On the Ubuntu runner used for the frozen standard evidence job, estimator-side numerical latency averages 16.60 ms, while completion-stop latency averages 13.26, 13.89, and 14.89 ms. These are CPU numerical-fixture timings and do not establish real GPU speedup.

For all three tasks,

\[
\texttt{unsafe\_act\_on\_covered\_episode\_count}=0.
\]

That mechanical result is expected from Proposition 2 for the matched numerical task readers; it is not an independent robot-safety result.

### 8.3 Distribution-shift stress

Under two-fold sensor noise, observed trajectory-envelope coverage is 93.33% in the 60-episode stress sample. This is an empirical readout only; the in-distribution exchangeability claim is not transferred to the shifted generator.

| Task | \(k_C<k_E\) | Estimator completion | Completion-stop completion | HOLD | Unsafe ACT |
|---|---:|---:|---:|---:|---:|
| Top suction | 95.00% | 100.00% | 98.33% | 1.67% | 0.00% |
| Label alignment | 68.33% | 98.33% | 98.33% | 1.67% | 0.00% |
| Keyed insertion | 0.00% | 98.33% | 0.00% | 100.00% | 0.00% |

The keyed-insertion result is intentionally retained. Under the shifted readout, the completion set never becomes sufficiently narrow for the coupled insertion reader, so the method refuses to ACT.

---

## 9. What the Results Establish

The numerical evidence supports four bounded conclusions.

First, latent pose identity is not necessary for the implemented task readers. A set of unresolved poses can be sufficient when the task verdict is invariant across the set.

Second, a naive calibrated set can be too conservative to be operationally useful. Coverage and usefulness are separate design axes.

Third, a learned observable shape followed by independent trajectory-level calibration can recover useful task certificates on this numerical backend: \(k_C<k_E\) occurs frequently while completion remains close to estimator-side stopping.

Fourth, the benefit is task dependent and can disappear under distribution shift, with HOLD acting as an explicit failure mode rather than being removed from the denominator.

The evidence does not establish that the learned envelope is optimal, that the observed 85% final-test coverage equals the nominal target, or that the same behavior transfers to RGB-D perception.

---

## 10. Reproducibility and Evidence Discipline

The repository is public and executable with one command:

```bash
python -m pip install -r requirements.txt
python reproduce.py --profile standard
```

GitHub Actions reruns the numerical stack on Ubuntu, macOS, and Windows. A separate standard-evidence job executes the larger TRAIN/CALIBRATION/FINAL TEST experiment and stores artifacts.

The project applies the glosa five-question discipline:

1. what was actually seen/run;
2. what the record alone separates;
3. what AI supplied;
4. what assumptions were introduced;
5. what independent mechanical/external check was performed.

The public claim card records AI contribution, split lineage, evidence boundaries, and the absence of external human replication.

Current evidence ceiling:

\[
\boxed{K1\;\text{public provisional / I4 mechanical-original-record}.}
\]

The project does not promote CI reproducibility into external empirical replication.

---

## 11. Limitations and Next Experiment

The current backend is numerical ICP on generated point clouds. It is not a camera pipeline.

The learned error-shape model is deliberately simple and fitted to one synthetic generator. Its feature choice was developed before the frozen final seeds were opened, but it is not claimed to be optimal or transferable.

The first frozen final test observes 85% trajectory coverage against a nominal 90% target. A next statistical step should repeat complete TRAIN/CALIBRATION/FINAL TEST cycles across independent seeds to estimate the empirical distribution of realized coverage and stopping utility rather than relying on one calibration realization.

The decisive robotics experiment then requires one real iterative 6D backend. A strong design would expose intermediate refinement states from the same initial RGB-D observation and compare:

1. fixed/full refinement;
2. estimator-side early stopping;
3. coverage-qualified task stopping.

The real gate must be calibrated on data disjoint from final task evaluation. The downstream outcome must be measured independently of the stopping gate.

A real speed claim requires

\[
T_{total}^{completion}<T_{total}^{estimator},
\]

including the cost of uncertainty features, shape-model inference, calibration adapter, and gate evaluation.

Physical task completion must be evaluated with HOLD retained in the denominator.

---

## 12. Conclusion

Estimator convergence and downstream task sufficiency are different stopping conditions.

This paper makes that distinction operational under incomplete state knowledge. Rather than asking whether one pose estimate is sufficiently confident, the proposed gate asks whether the entire calibrated set of pose errors still compatible with the retained evidence has collapsed to one downstream task verdict.

The resulting principle is:

\[
\boxed{
\text{Stop not when the pose is fully resolved, but when the unresolved set is task-invariant.}
}
\]

The public numerical experiment shows both the promise and the cost of that requirement. A naive high-coverage set can be uselessly conservative. A learned-shape calibrated set can stop earlier on most in-distribution episodes while preserving similar task completion in the numerical fixture, but its first frozen final test observes 85% rather than nominal 90% whole-trajectory coverage. Under shifted noise, one coupled task becomes completely non-actionable.

Those results narrow the next question rather than close it. The remaining test is whether a real iterative RGB-D 6D pose backend can construct a sufficiently informative calibrated completion envelope early enough to reduce end-to-end perception time without unacceptable loss of physical task completion.

---

## References

Angelopoulos, A. N., & Bates, S. (2021/2022). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.

Hibbard, M., Tanaka, T., & Topcu, U. (2023). Simultaneous perception–action design via invariant finite belief sets. *Automatica, 155*, 111140. https://doi.org/10.1016/j.automatica.2023.111140

Hietanen, A., Latokartano, J., Foi, A., Pieters, R., Kyrki, V., Lanz, M., & Kämäräinen, J.-K. (2021). Benchmarking pose estimation for robot manipulation. *Robotics and Autonomous Systems, 143*, 103810. https://doi.org/10.1016/j.robot.2021.103810

Ma, Y., Pakdamansavoji, S., Eret, C., Yang, R. H., Zhao, X., Zhang, Y., Cao, T., & Rasouli, A. (2026). *AnyBox: Efficient Zero-Shot 9DoF Pose Estimation of Boxes for Robotic Manipulation*. arXiv:2511.15884.

Mitash, C., Shome, R., Wen, B., Boularias, A., & Bekris, K. (2020). Task-Driven Perception and Manipulation for Constrained Placement of Unknown Objects. *IEEE Robotics and Automation Letters, 5*(4), 5605–5612. https://doi.org/10.1109/LRA.2020.3006816

Naik, L., Iversen, T. M., Kramberger, A., & Krüger, N. (2025). Robotic task success evaluation under multi-modal non-parametric object pose uncertainty. *Industrial Robot, 52*(5), 651–662. https://doi.org/10.1108/IR-10-2024-0467

Wang, J., Li, Z., Wang, Z., Guan, B., Shang, Y., & Yu, Q. (2025). Deterministic Object Pose Confidence Region Estimation. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 14866–14875.

Yang, H., & Pavone, M. (2023). Object Pose Estimation with Statistical Guarantees: Conformal Keypoint Detection and Geometric Uncertainty Propagation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 8947–8958. https://doi.org/10.1109/CVPR52729.2023.00864
