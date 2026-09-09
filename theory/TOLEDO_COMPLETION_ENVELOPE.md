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

## Pose-specific executable envelope

The current public harness represents unresolved local pose error by a symmetric box

\[
|e_i|\le b_i.
\]

It enumerates the box vertices and applies the declared downstream task reader to each vertex. This is exact for the monotone absolute-value toy readers currently implemented (coordinate box tolerances and the keyed-insertion L1 reader), but is not asserted to be exact for arbitrary robot tasks.

This construction creates a direct test of the distinction:

```text
pose not known exactly
            !=
task verdict unresolved
```

A wide yaw envelope can remain PASS for top suction while producing HOLD for label alignment. Shrinking only yaw can move label alignment from HOLD to PASS.

## Claim boundary

- The equations C1–C6 are **new definitions/derivations for this project**.
- They are rooted in Toledo's retained-state, domain-weld, reader-preservation, and reader-equivalence anchors.
- They do not establish a probability distribution over the latent pose.
- They do not convert ignorance into stochastic ontology.
- A finite completion set is a retained computational readout, not the hidden world.
- Real coverage, transition accuracy, and manipulation validity remain empirical questions.
