# Real-Backend Protocol v1.1 — University Lab Test

Implementation entry point: [`../lab/README.md`](../lab/README.md).

## Single testable claim

For an iterative 6D pose backend, define the certificate hitting time

\[
k_C=\inf\{k:\mathcal C_k^{cal}\subseteq\mathcal A_T\},
\]

where `C_k^cal` is the calibrated pose-error completion envelope and `A_T` is the declared task-admissible set. If no certificate occurs within budget, `k_C=+infinity` conceptually and the implementation reports no certificate rather than substituting the terminal stage.

Let `k_E` be estimator-side stopping. The real-system question is

\[
\boxed{k_C<k_E?}
\]

followed by two independent empirical questions:

1. Does the certificate policy preserve task-admissibility completion within a predeclared non-inferiority margin?
2. When the policies are executed online, does certificate stopping reduce measured latency after all policy-specific overhead?

## Experimental roles

Use disjoint TRAIN, CALIBRATION, and FINAL TEST data.

TRAIN fits the mapping from online-observable refinement features to pose-error shape. CALIBRATION freezes one whole-trajectory conformal score per independent episode. FINAL TEST is opened only after backend, features, alpha, task reader, comparator definitions, statistical plan, and timing plan are frozen.

## Certificate correctness

The reference implementation uses the finite split-conformal augmented order statistic. For

\[
r=\lceil(n+1)(1-\alpha)\rceil,
\]

if `r=n+1`, the retained quantile is `+infinity`. The resulting envelope is uninformative and the system fails closed to CONTINUE/HOLD. The implementation does not cap the rank at `n`.

Non-finite scores, invalid model coefficients/features, malformed task tolerances, and numerical overflow must never create a narrow certificate. Corrupt certificate state is treated as unbounded uncertainty.

## Required backend instrumentation

At every refinement stage export:

- stage index `k`;
- online-observable feature vector;
- incremental trajectory-prefix computation time;
- native estimator-stop flag.

Independently record the 6D relative pose error from a predeclared reference. Translation is metres. Rotation is represented by relative rotation-vector components in radians; these components are not globally interchangeable with Euler roll/pitch/yaw.

## Comparative design

At minimum compare estimator-side stopping and the coverage-qualified certificate. For a contribution claim, include matched alternatives on the same frozen splits when feasible:

1. full configured budget;
2. fixed budget tuned on development/calibration only;
3. estimator/residual threshold tuned on development/calibration only;
4. learned scalar task-score threshold using comparable observable features and risk control;
5. raw calibrated envelope ablation;
6. trajectory-conformal completion certificate.

Comparator parameters must be frozen before FINAL TEST outcomes are opened.

## Completion inference

Predeclare:

- non-inferiority margin;
- confidence level;
- independent sampling unit;
- minimum sample size / power rationale.

The public lab evaluator uses paired binary discordances and a conservative exact lower confidence bound based on Clopper–Pearson bounds with Bonferroni allocation. Point equality or a degenerate bootstrap interval is not sufficient. If the declared minimum sample size is not reached, non-inferiority remains unestablished.

## Timing

Two timing modes are distinguished.

### Trajectory-prefix estimate

A full refinement trajectory is generated and the recorded stage costs are summed only to each counterfactual endpoint. This is useful descriptive instrumentation but does not establish online policy speedup.

### Online-policy measured

For a latency claim, execute certificate-stop and estimator-stop loops separately under a predeclared paired timing design. Include feature extraction, refinement updates, certificate/gate cost, synchronization, and other policy-specific overhead. Use device-appropriate synchronization and warm-up.

Only direct online-policy timing licenses a latency-reduction decision.

## Required output

Report:

- held-out whole-trajectory envelope coverage;
- certificate rate;
- `k_C<k_E` rate across all assigned episodes;
- `k_C` distribution conditional on a certificate;
- executed endpoint distribution including terminal HOLD;
- estimator-stop distribution;
- overall task completion with HOLD in denominator;
- HOLD rate;
- unsafe ACT rate;
- unsafe ACT on covered episodes;
- paired completion difference and non-inferiority lower bound;
- trajectory-prefix timing as descriptive output;
- direct online-policy timing when making a speed claim.

## Real-backend claim

Permitted when real sensor/backend trajectories use independent pose ground truth and actual backend execution. Example:

> On backend X under protocol Y, the coverage-qualified certificate occurred before estimator-side stopping in Z% of held-out episodes; completion difference and its predeclared non-inferiority bound were E; direct paired online timing was D.

Do not substitute trajectory-prefix timing for D.

## Physical manipulation claim

Pose-ground-truth replay does not establish physical task non-inferiority. A physical manipulation claim requires actual robot executions with independently measured task outcomes under a predeclared randomized or blocked design.

## Distribution shift

Split-conformal interpretation is conditional on the declared exchangeability setting. Camera, object family, lighting, noise, backend version, or initialization shifts must be reported separately and may require recalibration. A shifted condition that produces HOLD is not automatically an OOD detector.

## Falsifiers

The proposal is unsupported for a tested backend/task if, under the declared operating point:

- certificates rarely occur before estimator stopping;
- held-out coverage is inadequate;
- completion fails the predeclared non-inferiority criterion;
- HOLD is operationally unacceptable;
- unsafe ACT is unacceptable;
- direct online-policy latency does not decrease after overhead;
- realistic shifts produce confident wrong certificates.

## One-command evaluator

```bash
python lab/run_real_system.py \
  --config lab/config.example.json \
  --train train.jsonl \
  --calibration calibration.jsonl \
  --test test.jsonl \
  --out lab_results.json
```

The evaluator is backend-agnostic. A university lab can test FoundationPose, ICP, a learned iterative refiner, or another 6D system by adapting the data-export layer.
