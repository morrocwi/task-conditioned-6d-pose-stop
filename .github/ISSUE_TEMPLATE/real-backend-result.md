---
name: Real backend result
title: "[REAL BACKEND] <backend> / <task> / <lab>"
about: Submit a supporting or falsifying real iterative 6D pose result
labels: ''
assignees: ''
---

## Laboratory / team

Institution or team:

Independent from repository author? `yes / no / mixed`

## Backend

- Backend:
- Commit/version:
- Weights/model identifier:
- Sensor:
- Compute hardware:
- Ground-truth reference:

## Frozen protocol

- Repository commit/tag:
- TRAIN episodes:
- CALIBRATION episodes:
- FINAL TEST episodes:
- Independent sampling unit:
- Alpha / nominal coverage:
- Task reader:
- Non-inferiority margin:
- Confidence level:
- Minimum sample-size rationale:
- Timing mode: `trajectory_prefix_estimate / online_policy_measured`

## Machine-readable result

Attach or link `lab_results.json` produced by `lab/run_real_system.py`.

## Primary readouts

- Held-out trajectory coverage:
- Certificate rate:
- `k_C < k_E` rate:
- Completion difference:
- Non-inferiority lower bound / decision:
- HOLD rate:
- Unsafe ACT rate:
- Online latency difference (only if directly measured):

## Distribution shifts / failures

Describe occlusion, object, initialization, sensor, lighting, or other shifts tested and any confident-wrong certificates or excessive HOLD.

## Claim boundary

What does this result **not** establish?

## Reproduction material

Provide code/data/archive references and access conditions.
