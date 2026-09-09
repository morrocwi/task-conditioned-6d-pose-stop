# Contributing Real-Backend Evidence

This project welcomes **falsifying as well as supporting** results from external laboratories.

The preferred contribution is not a new prose claim. It is a reproducible result package produced from a real iterative 6D pose backend under the public lab protocol.

## Minimum contribution package

Please include:

```text
lab_results.json
backend_and_hardware.md
config.json
split_manifest.json
exact commit/tag of this repository
exact backend commit/model weights
```

Raw sensor data may remain at an institutional archive if licensing/privacy/storage prevents inclusion in this repository, but the archive identifier and access conditions should be stated.

## Required backend information

Report:

- backend name and version/commit;
- model/weights identifier;
- camera/sensor and calibration method;
- compute hardware;
- operating system, CUDA/runtime/library versions when relevant;
- object/task set;
- independent ground-truth method;
- estimator-side stopping rule;
- online observable features;
- timing/synchronization method.

## Experimental lineage

Keep TRAIN, CALIBRATION, and FINAL TEST disjoint and freeze the method before opening FINAL TEST.

Predeclare:

- alpha / nominal coverage;
- task reader and tolerances;
- non-inferiority margin;
- confidence level;
- minimum sample size / power rationale;
- independent sampling unit;
- timing mode.

If the method is changed after FINAL TEST is inspected, start a new test lineage and retain the previous result.

## Timing claims

`trajectory_prefix_estimate` is acceptable for debugging and counterfactual analysis but does **not** support a speed claim.

For a latency claim, provide direct separately executed paired policy timing with

```text
timing_mode = online_policy_measured
```

and include policy-specific feature extraction, updates, gate/certificate cost, synchronization and relevant device timing.

## Negative results are first-class contributions

Please submit results when:

- coverage misses the declared target;
- the certificate is too conservative and HOLDs excessively;
- a scalar or estimator threshold dominates the certificate;
- direct online timing shows no saving;
- distribution shift produces confident wrong certificates;
- the task reader does not predict physical task outcome.

Do not omit these because they conflict with the current hypothesis.

## Cross-lab report

Use [`lab/RESULT_TEMPLATE.md`](lab/RESULT_TEMPLATE.md) and the machine-readable output from [`lab/run_real_system.py`](lab/run_real_system.py).

A real-sensor/backend result supports only the claim measured. Physical manipulation requires actual robot execution with independent physical task outcomes.

## Evidence status

External contributions should identify whether the execution was performed by an independent laboratory/team. Repository maintainers will not label an external result as independently replicated unless its authorship and execution provenance support that status.
