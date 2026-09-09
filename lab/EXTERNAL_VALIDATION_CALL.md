# Call for Independent Real-Backend Validation

The current repository intentionally stops at public numerical evidence. We invite university and robotics laboratories to test the method on a **real iterative 6D pose backend**.

The experiment is designed to be falsifiable. A useful contribution may show that the certificate works, fails, is too conservative, offers no latency benefit, or breaks under shift.

## Minimal research question

Does the calibrated task certificate occur before estimator stopping,

\[
k_C<k_E,
\]

while held-out task-admissibility completion satisfies the predeclared non-inferiority criterion and directly measured online-policy latency is reduced?

## Suggested first systems

Any iterative pose system is eligible. FoundationPose is an obvious contemporary target, but no backend is privileged by the method.

## Minimum evidence

A strong real-backend test should contain:

- real sensor trajectories;
- independent 6D pose reference;
- disjoint TRAIN/CALIBRATION/FINAL TEST roles;
- a frozen task reader;
- estimator-side comparator;
- matched simpler comparator(s);
- direct online-policy timing if claiming speed;
- occlusion/initialization/object or other declared shifts;
- complete reporting of HOLD and unsafe ACT.

Use `lab/run_real_system.py`, `lab/RESULT_TEMPLATE.md`, and the real-backend issue template.

Negative results are explicitly welcome and should remain in the public evidence record.
