# From trajectory coverage to a task stopping certificate

This note states the statistical bridge used by the learned-shape completion experiment. It is **not an existing Toledo theorem**.

## Inputs

Let a frozen observable shape model produce positive coordinate scales `s_{k,i}` at refinement stage `k`. Hidden pose errors are `e_{k,i}` and are unavailable to the online gate.

For one calibration episode define

\[
A=\max_{k\le K}\max_{i\le6}\left[\log(|e_{k,i}|+\delta_i)-\log s_{k,i}\right].
\]

Let `q_hat` be the ordinary split-conformal order statistic computed from independent calibration episodes at target miscoverage `alpha`.

The online completion envelope is

\[
\widehat{\mathcal C}^{cal}_k
=\left\{e:\ |e_i|\le \exp(\log s_{k,i}+\widehat q)-\delta_i\quad\forall i\right\}.
\]

The learned shape model is fitted before calibration and is treated as fixed when conformal calibration is applied.

## Proposition P1 — whole-trajectory marginal containment

**[NEW–DERIVATION/PROPOSAL; conditional on the standard split-conformal exchangeability assumptions]**

For a future episode exchangeable with the calibration episodes,

\[
\boxed{
\Pr\left(
 e_k^{true}\in\widehat{\mathcal C}^{cal}_k
 \quad\forall k\le K
\right)\ge 1-\alpha.
}
\]

### Reason

The event

\[
A_{new}\le\widehat q
\]

is algebraically equivalent to simultaneous containment of all six hidden pose-error coordinates at all retained stages. Standard split conformal prediction gives marginal coverage for the scalar episode score `A_new`. No optional-stopping argument is needed because calibration is attached to the whole trajectory before an adaptive stopping stage is selected.

This proposition imports the statistical guarantee from split conformal prediction; Toledo contributes only the retained-state/reader discipline used to interpret what the resulting set may license.

## Proposition P2 — covered robust ACT implies numerical task PASS

**[NEW–DERIVATION/PROPOSAL]**

Let `O_T` be the declared numerical task reader. Suppose the gate returns ACT only when

\[
O_T(e)=PASS\qquad\forall e\in\widehat{\mathcal C}^{cal}_k.
\]

Then on any episode for which the true hidden error is contained in the envelope at every stage,

\[
\boxed{
ACT_k\Rightarrow O_T(e_k^{true})=PASS.
}
\]

This is set inclusion, not probability: if the true error is a member of a set whose every member passes the reader, the true error passes that same reader.

Combining P1 and P2 yields the conditional numerical implication

\[
\boxed{
\Pr(\text{unsafe ACT caused by pose-reader violation})\le\alpha,
}
\]

provided that:

1. calibration and deployment episodes satisfy the exchangeability conditions required by split conformal prediction;
2. the online shape model and conformal scale are frozen before final evaluation;
3. the completion-set construction used by the gate matches the containment event used in calibration;
4. `O_T` is the correct downstream reader for the outcome being claimed.

## Non-collapse

The final inequality does **not** mean:

- physical robot failure probability is at most `alpha`;
- collision risk is at most `alpha`;
- the pose distribution is known;
- the conformal set is conditionally calibrated for every object, view, or context;
- the guarantee survives arbitrary distribution shift.

It bounds only the declared numerical pose-reader failure mechanism under the stated assumptions.

In glosa terms, the chain must remain explicit:

```text
observable readout
-> frozen shape model
-> calibrated completion envelope
-> task-reader invariance
-> numerical ACT/HOLD
-> independently evaluated outcome
```

No node is silently promoted into the next.

## Mechanical check

`tests/test_learned_conformal_completion.py` checks the code-level implication that, for the implemented monotone task readers, any hidden pose error lying inside a robust-PASS box also passes the same numerical outcome reader. The experiment additionally reports `unsafe_act_on_covered_episode_count`, which must remain zero.

## External statistical basis

- Angelopoulos, A. N., & Bates, S. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.

Pose uncertainty sets calibrated with conformal methods are prior art; this project does not claim conformal pose uncertainty as novel. The research question is whether a calibrated pose completion set can serve as a **downstream task stopping certificate** for iterative refinement.
