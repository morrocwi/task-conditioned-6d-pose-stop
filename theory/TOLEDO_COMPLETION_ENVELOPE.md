# Toledo-compatible completion envelope for unresolved pose state

This note adds a **new domain construction** to the project. It does not add equations to Toledo itself and does not claim that the equations below are existing Toledo mathematics.

## Existing Toledo anchors

**[EXISTING–TOLEDO: weld/M.01.v1]**

\[
S_{n+1}=F(S_n,u_n,c_n,T_n).
\]

**[EXISTING–TOLEDO: weld/M.02.v1]**

\[
q_D(F(z,u,c,T))=F_D^{\#}(q_D(z),u,c,T),
\]

and reader preservation

\[
O_D(z;Q,c)=O_D^{\#}(q_D(z);Q,c).
\]

**[EXISTING–TOLEDO: weld/M.03.v1]**

\[
z\sim_{Q,O,c,L}z'
\iff
O(F^kz)=O(F^kz')\quad\forall k\le L.
\]

These anchors permit two latent states to remain different while being indistinguishable to a declared reader over a declared finite horizon.

---

## New construction

### Eq. C1 — admissible completion set

**[NEW–DEFINITION]**

For a retained readout `r_k`, context `c`, and declared admissibility constraints `A`, define

\[
\boxed{
\mathcal C_k(r_k,c,A)
=
\{z:\;q_D(z)=r_k\;\text{or remains compatible with }r_k,\;A(z,r_k,c)=1\}.
}
\]

`C_k` is not the set of states that are true. It is the set of latent completions that the present record has not yet excluded under the declared construction.

The informal word **imagination** refers only to enumerating or constructing members of this set. It is not evidence that any particular member exists.

### Eq. C2 — task-reader image

**[NEW–DERIVATION/PROPOSAL]**

Let `O_T` be the downstream task reader. Map every admissible completion through that reader:

\[
\boxed{
\mathcal Y_{T,k}
=
\{O_T(z):z\in\mathcal C_k\}.
}
\]

For a binary task reader, `Y_T,k` is a subset of `{PASS, FAIL}`.

### Eq. C3 — robust task verdict under incomplete knowledge

**[NEW–DERIVATION/PROPOSAL]**

\[
\boxed{
V_{T,k}=
\begin{cases}
PASS,&\mathcal Y_{T,k}=\{PASS\},\\
FAIL,&\mathcal Y_{T,k}=\{FAIL\},\\
HOLD,&\{PASS,FAIL\}\subseteq\mathcal Y_{T,k}.
\end{cases}
}
\]

This is the central extension. The model does not require latent certainty. It requires **reader-invariance across the uncertainty that remains admissible**.

Equivalently, for all `z,z'` in the completion set,

\[
\boxed{
z\sim_T z'\quad\text{whenever}\quad O_T(z)=O_T(z').}
\]

A task-conditioned stop is licensed only when the completion set lies inside one task-reader equivalence class.

### Eq. C4 — task ambiguity / imagination diameter

**[NEW–DEFINITION]**

For a metric `d_T` on task readouts,

\[
\boxed{
W_T(\mathcal C_k)
=
\sup_{z,z'\in\mathcal C_k}
 d_T(O_T(z),O_T(z')).
}
\]

For a binary PASS/FAIL reader with distance 0 for equal verdicts and 1 otherwise,

\[
W_T\in\{0,1\}.
\]

`W_T=0` means the unresolved latent alternatives no longer matter to this task reader. It does **not** mean the latent pose is known exactly.

### Eq. C5 — refinement value as guaranteed reader contraction

**[NEW–DERIVATION/PROPOSAL]**

Let `Phi_a(C_k)` be the set of completion sets that may remain after perception action `a`. Define the guaranteed task-reader contraction

\[
\boxed{
G_T(a\mid\mathcal C_k)
=
W_T(\mathcal C_k)
-
\sup_{\mathcal C'\in\Phi_a(\mathcal C_k)}W_T(\mathcal C').
}
\]

A cost-sensitive extension can rank actions by

\[
\boxed{
J_T(a)=\frac{G_T(a\mid\mathcal C_k)}{c(a)}
}
\]

when `c(a)>0`.

This does not claim that the transition model `Phi_a` is true. Its empirical adequacy must be tested separately.

### Eq. C6 — finite executable approximation

**[NEW–DEFINITION]**

A computer cannot enumerate an unrestricted latent world. The executable harness therefore uses a finite retained completion set

\[
\boxed{
\widehat{\mathcal C}_k^{(M)}
=
\{z_k^{(1)},\ldots,z_k^{(M)}\}.
}
\]

The resulting verdict is only a verdict over the retained finite set:

\[
\widehat V_{T,k}^{(M)}
=V_T(\widehat{\mathcal C}_k^{(M)}).
\]

**Non-collapse:**

\[
\boxed{
\widehat V_{T,k}^{(M)}=PASS
\not\Rightarrow
V_{T,k}=PASS
}
\]

unless the construction of the finite set has an independently justified coverage guarantee.

---

## Calibration-derived completion envelope

The previous equations say what a completion set *means*. They do not yet say how a numerical system should obtain one from an imperfect observable proxy. The next construction adds that bridge.

The statistical basis is split conformal prediction: under exchangeability, a calibration-set order statistic can provide finite-sample marginal coverage for a new exchangeable unit. This project uses one score per **whole refinement episode**, rather than one score per adaptively selected stage, so that coverage is attached to the retained trajectory as the unit being calibrated. This statistical result is external to Toledo; the equations below are project-specific applications.

### Eq. C7 — trajectory nonconformity score

**[NEW–DEFINITION; EXTERNAL-METHOD BASIS: split conformal prediction]**

Let `p_{e,k,i}` be the observable uncertainty proxy for episode `e`, refinement stage `k`, and local pose coordinate `i`. Let `epsilon_i>0` be a declared numerical floor. During offline calibration only, let the hidden evaluation error be `e_{e,k,i}`. Define one score per episode:

\[
\boxed{
A_e
=
\max_{k\le K}\max_{i\le 6}
\frac{|e_{e,k,i}|}{p_{e,k,i}+\epsilon_i}.
}
\]

The maximum over stages is deliberate. It avoids silently treating the stage chosen later by an adaptive stopping rule as though it were an independently calibrated sample.

### Eq. C8 — frozen calibration scale

**[NEW–DERIVATION/PROPOSAL; EXTERNAL-METHOD BASIS: split conformal order statistic]**

For `n` exchangeable calibration episodes and target miscoverage `alpha`, sort the episode scores

\[
A_{(1)}\le\cdots\le A_{(n)}
\]

and define

\[
\boxed{
\widehat q_{1-\alpha}
=
A_{(r)},
\qquad
r=\min\{n,\lceil(n+1)(1-\alpha)\rceil\}.
}
\]

The scale is frozen before held-out evaluation.

### Eq. C9 — calibrated trajectory completion box

**[NEW–DERIVATION/PROPOSAL]**

At deployment stage `k`, the gate sees the observable proxy but not hidden pose error. It constructs axis bounds

\[
\boxed{
b_{k,i}
=
\widehat q_{1-\alpha}(p_{k,i}+\epsilon_i)
}
\]

and the retained completion box

\[
\boxed{
\widehat{\mathcal C}^{\,cal}_k
=
\{e:\;|e_i|\le b_{k,i}\;\forall i\}.
}
\]

Under the exchangeability assumptions required by split conformal prediction, the episode-level construction targets marginal coverage of the complete retained trajectory, not certainty about the hidden world.

### Eq. C10 — coverage-qualified task stop

**[NEW–DERIVATION/PROPOSAL]**

The online task decision is

\[
\boxed{
D_{T,k}=
\begin{cases}
ACT,&O_T(e)=PASS\quad\forall e\in\widehat{\mathcal C}^{cal}_k,\\
CONTINUE,&\text{otherwise and refinement remains available},\\
HOLD,&\text{otherwise}.
\end{cases}
}
\]

The decisive non-collapse is

\[
\boxed{
\text{conformal marginal coverage}
\neq
\text{deterministic safety guarantee}.
}
\]

Distribution shift, non-exchangeability, an inadequate proxy, or an incorrect downstream task reader can invalidate the practical meaning of the completion envelope even when the code implements the order statistic correctly.

---

## Pose-specific executable envelope

The public harness now contains two implementations.

1. `theory/completion_envelope.py` uses a manually declared symmetric box and finite vertex enumeration.
2. `experiments/conformal_completion.py` derives the box scale from disjoint trajectory-level calibration episodes and evaluates the frozen envelope on held-out numerical ICP/Kabsch trajectories.

For the monotone absolute-value readers currently implemented, robust PASS can be checked exactly from the axis bounds:

- box task: every task-relevant bound is within its tolerance;
- keyed-insertion L1 task: the sum of normalized task-relevant bounds is at most one.

This creates a direct test of the distinction:

```text
pose not known exactly
            !=
task verdict unresolved
```

A wide yaw envelope can remain PASS for top suction while producing HOLD for label alignment. Shrinking yaw can move label alignment from HOLD to PASS.

## Claim boundary

- Equations C1–C10 are **new definitions/derivations for this project**.
- They are rooted in Toledo's retained-state, domain-weld, reader-preservation, and reader-equivalence anchors.
- C7–C10 additionally use split-conformal calibration as an external statistical method; conformal prediction is not a Toledo result and is not claimed as novel here.
- The completion envelope does not establish a probability distribution over latent pose states.
- It does not convert ignorance into stochastic ontology.
- A calibrated finite envelope is a retained computational readout, not the hidden world.
- Exchangeability is an assumption that must be stress-tested rather than presumed.
- Real RGB-D coverage, transition accuracy, manipulation validity, and physical safety remain empirical questions.

## External statistical reference

Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
