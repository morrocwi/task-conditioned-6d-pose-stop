# Predeclaration — real-bop-lmo-2026-09-09-run6 (PROP-NATIVE-03, memory/persistence accumulator)

Sixth real-data cycle. Founder authorization (`ops/HANDOFF_2026-09-09_external_dataset.md`,
"Run 6 authorization", 2026-09-09: "เอาเลยรอบที่หก"). Follows run5's `refuted_unsafe` finding for
PROP-NATIVE-02 (the single-instant invariance check ACTs at k=0 on 100% of test episodes, wrong
92.5-100% of the time — root cause: the perturbation magnitude is far smaller than the real
coarse-detector initial-pose error, so the check answers "is a tiny wobble tolerable" instead of
"is this estimate correct").

## Toledo-first

`PROP-NATIVE-03` was already registered in
`~/ANSE.ASIA/toledo/registry/proposals/native_retained_sensitivity.json` before this cycle began
(status `unverified`). This run does not re-register it; its `status` field is updated with the
real empirical result after the run (see the handoff-file update and Toledo's own commit).

## Registered rule and a disclosed interpretation choice

Registered statement:

```
M_k := M_{k-1} + 1[task verdict invariant under every imagined perturbation at stage k],  M_0:=0
ACT <=> M_k >= theta
```

Read literally, this LaTeX is a non-resetting running sum. Toledo's own `note_ascii` for the same
entry describes it as "a streak count of **consecutive** stages" and the proposal's own name is
"a **persistent streak** of task-invariance, not a single-instant check". A non-resetting sum and a
resetting streak counter agree only until the first failure after a pass; the registered intent
(persistence/noise-robustness) is the resetting-streak reading, not the never-decreasing sum. This
run therefore implements the **resetting streak**:

```
M_k = M_{k-1} + 1   if the stage-k invariance check passes
M_k = 0             otherwise
```

Disclosed here rather than silently picked — see `lab/native_persistence.py`'s own docstring for
the same statement, in case a reviewer wants to check the code matches this document.

## Reused, unmodified from run5

- `lab/native_sensitivity.py:stage_decision` (PROP-NATIVE-02's single-instant invariance check),
  `NativeConfig`, `lambda_ref=0.33927484832749355`, `cap=0.03274911721483188` (both TRAIN-derived
  in run5, not re-derived here).
- `lab/data/bop_lmo_episodes_native/{train,calibration,test}.jsonl` — identical file, identical
  split (`split_manifest.json` diffed byte-identical against run5's copy before this commit).
- Task tolerances, `error_floor`, inference config (margin/confidence/minimum_n): byte-identical
  to run5's `config.json`.
- `lab/run_real_system.py`'s shared utility functions (`estimator_index`, `cumulative_ms`,
  `boot_mean_diff`, `wilson`, `ensure_disjoint`, `validate_dataset`, `load_jsonl`, `task_pass`) and
  `cqts/safety.py`'s `paired_binary_noninferiority`/`validate_task_spec` — unchanged, no
  reimplementation.

## What is genuinely new

`lab/native_persistence.py`: a resetting streak accumulator `M_k` wrapping the (unmodified)
per-stage `stage_decision` call, gating ACT on `M_k >= theta`. `lab/run_native_persistence.py`: a
standalone evaluator (mirrors `lab/run_native_sensitivity.py`'s structure) supporting two
predeclared variants:

- **variant (a)** — theta-gate only. Isolates the memory variable cleanly, per the handoff's
  stated preference to try (a) before (b).
- **variant (b)** — theta-gate AND a residual-scaled perturbation magnitude
  (`m_b = min(m_native * max(1, residual_rmse_k/residual_scale_ref), residual_cap_multiplier*CAP)`),
  addressing PROP-NATIVE-02's own diagnosed root cause (perturbation magnitude has no
  absolute-residual-scale term) directly, as a second, clearly separate change. Reported
  separately from (a), never conflated with it.

## Honest TRAIN-only diagnostic, computed and disclosed BEFORE test.jsonl was read for run6

`lab/derive_native_persistence_from_train.py` (reads ONLY `train.jsonl`, no calibration/test, no
ground truth anywhere — theta and the residual-scale reference are online-decision parameters, not
an external task-tolerance choice, so they are derived exactly like `lambda_ref`/`cap` were in
run5, from the observable per-stage indicator/feature distribution alone):

- PROP-NATIVE-02's single-instant check evaluates **True at every one of 640 TRAIN stage readings
  (40 episodes x 16 stages), for all three tasks** — `stage_invariant_true_rate = 1.0` at every
  stage, `initial_streak_length = 16/16` for every TRAIN episode, for all three tasks.
- **This predicts, before test.jsonl is opened, that variant (a) alone will reduce to a
  deterministic fixed-stage rule**: since the per-stage indicator has no TRAIN-observed variation
  to threshold against, `M_k` is deterministically `k+1` for every episode, and ACT fires at the
  fixed stage `k=theta-1` regardless of episode content — not a genuine noise filter, whatever the
  test-time numbers turn out to be. This is reported here as a predeclared prediction, not
  discovered after running against test.

## theta (predeclared)

`theta := 4`, chosen as the smallest value in run3's own already-frozen Bonferroni checkpoint set
`{4, 8, 12, 15}` (`lab/results/real-bop-lmo-2026-09-09-run3/config.json`) — an existing system
constant frozen for an unrelated purpose before run 6 was conceived, rather than a new arbitrary
round number invented for this cycle, while still small enough to satisfy the handoff's own stated
preference ("Prefer (a) first ... it isolates the memory variable cleanly").

## residual_scale_ref / residual_cap_multiplier (predeclared, variant (b) only)

- `residual_scale_ref := median(ICP residual RMSE)` over all 640 TRAIN stage readings
  (`features[1] = log(residual RMSE + 1e-9)`) = `0.018320411722306557`. Median chosen for the same
  robust-to-outlier reason run5's `lambda_ref` used a median.
- `residual_cap_multiplier := 2.0`, a predeclared doubling of run5's own `CAP`
  (`0.03274911721483188` -> effective ceiling `0.06549823442966376`) — the smallest round-number
  change that lets the residual-scale factor actually move the bound on stages where the current
  residual exceeds the TRAIN-typical value, rather than being immediately clipped back to run5's
  original `CAP`.
- Full derivation and every intermediate quantile: `native_persistence_threshold_derivation.json`
  in this directory.

## No test.jsonl access before this commit

`lab/run_native_persistence.py` was not invoked against `test.jsonl` before this commit
(`config.json`, `split_manifest.json`, `native_persistence_threshold_derivation.json`, this file,
and the new code in `lab/native_persistence.py`/`lab/run_native_persistence.py`/
`lab/derive_native_persistence_from_train.py`) was made. This matches the F4 process fix adopted
after run1 (frozen config in its own commit, strictly before the FINAL TEST evaluator is ever run
against `test.jsonl`).

## Same guardrails as runs 1-5 (not relitigated here)

Novelty/positioning constraints, no-AI-attribution, independent-review-before-origin-push,
TRAIN/CALIBRATION/FINAL-TEST discipline — all unchanged, see
`ops/HANDOFF_2026-09-09_external_dataset.md`.
