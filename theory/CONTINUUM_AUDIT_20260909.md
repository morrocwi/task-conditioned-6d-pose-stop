# Continuum-injection audit — task-conditioned-6d-pose-stop's formalization

Founder instruction (verbatim, 2026-09-09): "ต้องรื้อถอนอะไรที่เป็นคอนทินัมออกทั้งหมด เพราะในสมการเรา
ไม่มีสมการคอนทินัมแท้ มีแค่ปรากฏการเสมือนคอนทินัมเท่านั้น" (strip out everything continuum, because
our equations have no true continuum equation, only continuum-*appearing* phenomena) — "ตั้งเป็นหนี้
ที่ต้องไล่รื้อถอนด้วยนะ" (set this as a debt to track down and remove). This is now tracked as a
standing debt (workspace todolist "CONTINUUM-AUDIT"), not a one-off note.

Method: every equation in `theory/FORMALIZATION_v1.md` (C1–C26) and the new native proposals
(`~/ANSE.ASIA/toledo/registry/proposals/native_retained_sensitivity.json`, PROP-NATIVE-01/02)
checked against `information-discrete-math`'s contaminated-concept → discrete-replacement table.
**This audit does not yet fix everything it finds** — it names each injection, its severity, and
the discrete floor that should replace it, so the debt can be worked down item by item rather than
attempted all at once.

## Findings, most severe first

### 1. SE(3)/SO(3) itself as the pose representation — SEVERE, not yet fixed anywhere
`T* ∈ SE(3)`, `T̂_k ∈ SE(3)` (Part 0, and PROP-NATIVE-01/02's `T̂_k` too). `SO(3)` (3D rotation) is
a continuous Lie group in every formulation used so far — rotation matrices, the matrix
exponential/logarithm (`Log(...)`), or a rotation vector in `ℝ^3`. This is the direct 3D
generalization of the contaminated-concept table's own headline example: "angle/degree ... needs
ℝ-completeness = I1." Even the new native proposals (PROP-NATIVE-01/02), which fix the `T*`
non-readout, still represent `T̂_k` as a continuum-manifold element — **the continuum-pose
injection is NOT fixed by the native error redefinition alone.**
- **What a genuine discrete floor would look like:** a rotation restricted to a finite,
  rational-parametrized set (e.g. rational points on `SO(3)` via a Cayley transform over `ℚ`, or a
  fixed finite quantization grid the correspondence solver actually searches over) — never a
  literal continuum manifold silently assumed underneath a floating-point implementation.
- **Not attempted in this pass** — this is the largest, hardest item and needs its own dedicated
  design pass (likely its own Toledo root-extension proposal), not a quick patch.

### 2. The completion set `C_k` and its "possible worlds" `z` — SEVERE if read literally over `ℝ^6`
C1 (`C_k = {z : z remains compatible with r_k}`) and C4's `sup_{z,z'∈C_k}` (and C21's
`W̄_T(a|C_k)`) are written as if `C_k` ranges over a continuum subset of `ℝ^6`. In the actual
IMPLEMENTED system (C9–C10), `Ĉ^cal_k` is a finite-dimensional axis-aligned box `{e : |e_i|≤b_i}` —
already effectively discretized by the *task reader*'s own finite check, but the formal statement
of `C_k`/`W_T` itself still reads as continuum set theory (a `sup` over uncountably many `z`).
- **Discrete floor:** `W_T(C_k)` for a binary task reduces to checking finitely many *vertices* of
  the box (a task reader built from linear/box constraints is extremized at a corner), which is
  already effectively how `fail_closed_task_pass` computes it — the fix here is mostly *notational*
  (state `W_T` as a finite max over box corners, not a `sup` over a continuum set), not a new
  algorithm.

### 3. `log s_{k,i}` as a continuous log-linear regression (C6) — MODERATE
The observable-feature predictor is a smooth, real-coefficient linear model in log-space. Under
IDM's own stance, a real number is a legitimate *readout* of an underlying rational computation
(the regression coefficients, once fit, are rational/floating-point numbers — finite readouts), so
this is not a severe injection *by itself*. The injection risk is in the assumed *functional form*
(a smooth log-linear curve) being silently treated as the true underlying law rather than one
finite, fitted readout among others — this is exactly why `PROP-DECAY-01` (refuted, run 4) tried a
different functional form, and why `PROP-NATIVE-02` moves away from a fitted curve entirely toward
a directly-computed eigenvalue-based quantity.

### 4. Real-valued task tolerances `τ_i` and the region `A_T = {e : g_T(e)≤0}` (C11) — MODERATE
Tolerances are real numbers (millimetres, radians) — legitimate readouts of a declared engineering
requirement, not an injected continuum limit; this is closer to IDM's "a computed 0:ℚ inside an
identity is ordinary and fine" caveat than to a genuine I1–I4 injection. Flagged for completeness,
not treated as urgent.

### 5. `P(...) ≥ 1−α` marginal coverage (C16–C18) — LOW RISK, already well-handled
This LOOKS like a continuum-probability statement, but the actual construction (C7–C9) is a finite
exchangeability/order-statistic argument over `n` calibration episodes — a genuinely discrete,
combinatorial guarantee, not a continuum limit. **This is the one part of the existing
formalization that already matches IDM's own discipline well**, including its explicit refused
endpoint (`q_α = +∞` when `n` is too small — a textbook "approach, never reach" non-readout
boundary, not silently patched over). Worth noting as a positive existing example, not a debt.

### 6. Eigenvalues/eigenvectors of `H_k` (PROP-NATIVE-02) — LOW-MODERATE, tractable
Eigenvalues of a real symmetric matrix are, in general, algebraic numbers (roots of a real
polynomial) — not literally rational, but not a silently-injected continuum limit either, PROVIDED
`H_k`'s own entries are built from finite/rational input data (correspondence weights, coordinates)
rather than opaque floating-point accumulation. Today's real Coq proofs
(`InfoSpectralCeilingSharp.v`, `RDL_SpectralCeiling.v`) already show this workspace's own idiom for
handling exactly this case rationally: an eigenpair is taken *extensionally as a hypothesis*
(`form E x == lambda * norm2 n x`), never assuming general spectral theory over `ℝ`. PROP-NATIVE-02
should be built the same way if it advances past Dr tier — flagged for the implementer, not fixed
here.

## Priority order for closing this debt
1. Item 1 (SE(3)/SO(3) as a continuum manifold) — the deepest and most consequential; needs its own
   design pass, likely a new Toledo root-extension proposal (a discrete/rational parametrization of
   rotation), before any of items 2–6 can be called genuinely native rather than "native error
   definition wrapped around a still-continuum pose representation."
2. Item 2 (state `C_k`/`W_T` as finite box-corner checks, not continuum-set `sup`) — cheap,
   mostly notational, should be done alongside any future edit to `theory/FORMALIZATION_v1.md`.
3. Items 3–4 — lower urgency, already closer to IDM's "readout of the discrete" stance than to an
   injected non-readout.
4. Item 5 — no action needed, already compliant; kept here as the audit's own positive control.
5. Item 6 — apply the existing Coq idiom (extensional eigenpair hypothesis) if/when PROP-NATIVE-02
   moves past Dr tier.

This document does not close the CONTINUUM-AUDIT debt — it scopes it. Update this file (or its
Toledo counterpart) as each item above is actually addressed, rather than marking the whole debt
closed on the strength of this audit alone.
