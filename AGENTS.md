# AGENTS.md - task-conditioned-6d-pose-stop

## What this repository is

Public, implementation-first research repository asking a single question: can iterative 6D pose
refinement stop before estimator-side convergence because the entire calibrated pose-error
completion set is already acceptable for the declared downstream task? The online rule is:
observable refinement readout -> calibrated completion envelope -> robust downstream task reader
-> ACT | CONTINUE | HOLD. ACT is permitted only when every pose error retained by the envelope
passes the declared task reader. If no certificate exists within budget, the certificate hitting
time `k_C` is undefined / +infinity; a terminal HOLD at the budget is never relabelled as `k_C`.

`CLAIMS.md` is the standing status ledger: several real-sensor non-inferiority claims here are
refuted, several speed/non-inferiority claims are still open hypotheses, and none is confirmed -
read it before repeating any number from this repository.

## Read first

1. `README.md` - the core distinction, the online rule, and "What changed after adversarial review".
2. `CLAIMS.md` - every claim this repository makes, with its current status.
3. `RESEARCH_QUESTION.md` - the single question this repository tests.
4. `CONTRIBUTING.md` - what a reproducible falsifying or supporting result package must contain.

## Rules

These are rules the repository already states; this file adds none of its own.

- Pose not known exactly is not the same as task verdict unresolved; do not collapse the two.
- A terminal HOLD (budget exhausted, no certificate) is an executed endpoint, never relabelled as
  a certificate hit.
- Split-conformal marginal coverage is not a deterministic safety guarantee (`CLAIMS.md`).
- Keep TRAIN, CALIBRATION and FINAL TEST disjoint, and freeze the method before opening FINAL TEST.
- A contribution is a reproducible result package from a real backend, not a new prose claim
  (`CONTRIBUTING.md`).

## Programme map

This repository is one node of the Human-AI Readout Programme. Which repository answers which kind of
question, what to read first and which gate applies is kept in one place, the routing hub:
<https://github.com/morrocwi/main.hub> (start at its `AGENTS.md`, then `ROUTES.md`).
The hub holds pointers and pinned links only. It is a readout of one moment: when the hub and this
repository disagree, this repository wins.
