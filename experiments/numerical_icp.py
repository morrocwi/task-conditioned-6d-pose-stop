#!/usr/bin/env python3
"""Publicly reproducible numerical experiment for task-conditioned 6D pose stopping.

Evidence boundary
-----------------
EXECUTED: iterative point-to-point ICP/Kabsch registration on generated 3-D point clouds.
NOT EXECUTED: camera/RGB-D sensor, neural pose model, contact physics, physical robot.

The stopping gate never sees hidden ground-truth pose error. A task-specific threshold
is chosen on a tuning split using only observable ICP readouts, checked on a separate
safety-calibration split, then frozen before evaluation on held-out test seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TUNE_SEED = 2026090901
SAFETY_SEED = 2026090902
TEST_SEEDS = (2026090911, 2026090912, 2026090913)
STRESS_SEED = 2026090999


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], float)


def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th
    K = skew(k)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def rotvec(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-10:
        return np.zeros(3)
    s = math.sin(th)
    if abs(s) < 1e-10:
        vals, vecs = np.linalg.eig(R)
        axis = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        axis /= np.linalg.norm(axis)
        return axis * th
    v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2.0*s)
    return v * th


def compose(R1, t1, R2, t2):
    return R1 @ R2, R1 @ t2 + t1


def apply(R, t, P):
    return (R @ P.T).T + t


def inv(R, t):
    Ri = R.T
    return Ri, -Ri @ t


def pose_error(R, t, Rg, tg):
    """Evaluation-only 6-vector: tx,ty,tz,rx,ry,rz."""
    Ri, ti = inv(R, t)
    Re, te = compose(Ri, ti, Rg, tg)
    return np.r_[te, rotvec(Re)]


def best_fit(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A-ca).T @ (B-cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cb - R @ ca


def nearest(A, B):
    d2 = ((A[:,None,:] - B[None,:,:])**2).sum(2)
    idx = d2.argmin(1)
    return B[idx], np.sqrt(d2[np.arange(len(A)), idx])


def local_proxy(model, R, t, corr):
    """Local linearized dispersion proxy. Deliberately not called a posterior/CI."""
    X = apply(R, t, model)
    r = (X-corr).reshape(-1)
    J = np.vstack([np.hstack((-skew(x), np.eye(3))) for x in X])
    dof = max(1, len(r)-6)
    sigma2 = float((r@r)/dof)
    cov = sigma2 * np.linalg.pinv(J.T @ J, rcond=1e-10)
    sd = np.sqrt(np.maximum(0.0, np.diag(cov)))
    return np.r_[sd[3:], sd[:3]]


@dataclass
class Stage:
    k: int
    R: np.ndarray
    t: np.ndarray
    proposed_delta: np.ndarray
    proxy: np.ndarray
    rms: float
    dt_s: float


@dataclass
class Episode:
    stages: list
    Rg: np.ndarray
    tg: np.ndarray


def run_icp(model, obs, R0, t0, max_iter=20, damping=0.65):
    R, t = R0.copy(), t0.copy()
    out = []
    for k in range(max_iter+1):
        tick = time.perf_counter()
        X = apply(R, t, model)
        corr, dist = nearest(X, obs)
        proxy = local_proxy(model, R, t, corr)
        rms = float(np.sqrt(np.mean(dist**2)))
        if k == max_iter:
            out.append(Stage(k, R.copy(), t.copy(), np.zeros(6), proxy, rms, time.perf_counter()-tick))
            break
        dR, dtv = best_fit(X, corr)
        w = rotvec(dR)
        delta = np.r_[damping*dtv, damping*w]
        # State k, proxy and proposed next update are aligned before correction.
        out.append(Stage(k, R.copy(), t.copy(), delta, proxy, rms, time.perf_counter()-tick))
        R, t = compose(rodrigues(damping*w), damping*dtv, R, t)
    return out


def model_points(rng, n=120):
    P = rng.uniform([-0.035,-0.025,-0.018], [0.045,0.028,0.022], size=(n,3))
    P[:n//4,0] += 0.012
    P[n//4:n//3,2] += 0.006
    return P


def generate_episode(rng, max_iter, points, noise_scale=1.0):
    model = model_points(rng, points)
    wg = rng.normal(0.0, math.radians(18.0), size=3)
    Rg = rodrigues(wg)
    tg = rng.uniform([-0.03,-0.03,0.30], [0.03,0.03,0.38])
    obs = apply(Rg, tg, model) + rng.normal(0.0, 0.0008*noise_scale, size=model.shape)
    w0 = wg + rng.normal(0.0, [math.radians(6),math.radians(6),math.radians(18)])
    R0 = rodrigues(w0)
    t0 = tg + rng.normal(0.0, [0.008,0.008,0.010])
    return Episode(run_icp(model, obs, R0, t0, max_iter=max_iter), Rg, tg)


TASKS = {
    "top_suction": {
        "kind": "box",
        "mask": (0,1,2,3,4),
        "tol": np.array([0.005,0.005,0.008,math.radians(5),math.radians(5),math.inf]),
    },
    "label_alignment": {
        "kind": "box",
        "mask": (0,1,2,3,4,5),
        "tol": np.array([0.005,0.005,0.008,math.radians(5),math.radians(5),math.radians(4)]),
    },
    "keyed_insertion": {
        "kind": "l1",
        "mask": (0,1,5),
        "tol": np.array([0.004,0.004,math.inf,math.inf,math.inf,math.radians(3)]),
    },
}


def task_success(task, err):
    s = TASKS[task]
    if s["kind"] == "box":
        return bool(all(abs(float(err[i])) <= float(s["tol"][i]) for i in s["mask"]))
    return bool(sum(abs(float(err[i]))/float(s["tol"][i]) for i in s["mask"]) <= 1.0)


def observable_task_score(task, stage):
    """Lower is better. Uses only current local proxy and declared task tolerances."""
    s = TASKS[task]
    if s["kind"] == "box":
        return max(float(stage.proxy[i])/float(s["tol"][i]) for i in s["mask"])
    return sum(float(stage.proxy[i])/float(s["tol"][i]) for i in s["mask"])


def estimator_stop_stage(stages):
    prev_small = False
    for s in stages[:-1]:
        d = np.abs(s.proposed_delta)
        small = bool(np.all(d[:3] <= 0.0005) and np.all(d[3:] <= math.radians(0.1)))
        if small and prev_small:
            return s.k
        prev_small = small
    return stages[-1].k


def first_task_stage(ep, task, threshold):
    gate_time = 0.0
    for s in ep.stages:
        tick = time.perf_counter()
        ok = observable_task_score(task, s) <= threshold
        gate_time += time.perf_counter()-tick
        if ok:
            return s.k, True, gate_time
    return ep.stages[-1].k, False, gate_time


def cumulative_time(stages, k):
    return float(sum(s.dt_s for s in stages if s.k <= k))


def eval_threshold(ep, task, threshold):
    k, act, _ = first_task_stage(ep, task, threshold)
    if not act:
        return k, False, False
    st = ep.stages[k]
    return k, True, task_success(task, pose_error(st.R, st.t, ep.Rg, ep.tg))


def tune_threshold(episodes, task, min_act_rate=0.5):
    """Tune fastest threshold with zero observed unsafe ACTs on tuning data.

    The frozen threshold is then evaluated on a separate safety split and held-out test.
    """
    candidates = sorted({observable_task_score(task,s) for ep in episodes for s in ep.stages})
    best = None
    for th in candidates:
        rows = [eval_threshold(ep, task, th) for ep in episodes]
        n = len(rows)
        act = sum(a for _,a,_ in rows)
        unsafe = sum(a and not ok for _,a,ok in rows)
        if unsafe != 0 or act/n < min_act_rate:
            continue
        cand = (float(np.mean([k for k,_,_ in rows])), -(act/n), th)
        if best is None or cand < best:
            best = cand
    if best is None:
        return {"threshold": None, "status": "HOLD", "reason": "no zero-unsafe tuning threshold with required act rate"}
    return {"threshold": float(best[2]), "status": "TUNED", "tuning_mean_k": float(best[0]), "tuning_act_rate": float(-best[1])}


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = successes/n
    den = 1+z*z/n
    centre = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [max(0.0,centre-half), min(1.0,centre+half)]


def evaluate_frozen_threshold(episodes, task, threshold):
    rows = [eval_threshold(ep, task, threshold) for ep in episodes]
    n = len(rows)
    act = sum(a for _,a,_ in rows)
    unsafe = sum(a and not ok for _,a,ok in rows)
    success = sum(ok for _,_,ok in rows)
    return {
        "episodes": n, "act_rate": act/n, "hold_rate": 1-act/n,
        "completion_rate": success/n, "unsafe_act_rate": unsafe/n,
        "unsafe_act_wilson95": wilson(unsafe,n),
        "mean_k": float(np.mean([k for k,_,_ in rows])),
    }


def eval_policy(ep, task, policy, threshold, fixed_k=8):
    max_k = ep.stages[-1].k
    if policy == "full":
        k, act, gate_time = max_k, True, 0.0
    elif policy == "fixed8":
        k, act, gate_time = min(fixed_k,max_k), True, 0.0
    elif policy == "estimator_stop":
        k, act, gate_time = estimator_stop_stage(ep.stages), True, 0.0
    elif policy == "task_stop":
        if threshold is None:
            k, act, gate_time = max_k, False, 0.0
        else:
            k, act, gate_time = first_task_stage(ep, task, threshold)
    else:
        raise KeyError(policy)
    st = ep.stages[k]
    err = pose_error(st.R, st.t, ep.Rg, ep.tg)
    success = bool(act and task_success(task,err))
    return {
        "k": int(k), "act": bool(act), "hold": bool(not act), "success": success,
        "unsafe_act": bool(act and not success),
        "latency_s": cumulative_time(ep.stages,k)+gate_time,
        "trans_err_m": float(np.linalg.norm(err[:3])),
        "rot_err_deg": float(np.linalg.norm(err[3:])*180/math.pi),
    }


def bootstrap_diff(a, b, seed, repeats=3000):
    d = np.asarray(a,float)-np.asarray(b,float)
    rng=np.random.default_rng(seed); n=len(d); vals=np.empty(repeats)
    for i in range(repeats):
        idx=rng.integers(0,n,size=n); vals[i]=float(np.mean(d[idx]))
    return {"mean":float(np.mean(d)),"bootstrap95":[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}


def summarize(records, task):
    out={}
    for p in ("fixed8","full","estimator_stop","task_stop"):
        rs=[r[p] for r in records]; n=len(rs)
        success=sum(r["success"] for r in rs); act=sum(r["act"] for r in rs); unsafe=sum(r["unsafe_act"] for r in rs)
        ks=np.array([r["k"] for r in rs],float); lat=np.array([r["latency_s"] for r in rs],float)
        out[p]={
            "episodes":n,"mean_k":float(ks.mean()),"median_k":float(np.median(ks)),
            "completion_rate":success/n,"completion_wilson95":wilson(success,n),
            "act_rate":act/n,"hold_rate":1-act/n,"unsafe_act_rate":unsafe/n,
            "success_given_act":success/act if act else None,
            "mean_latency_ms":float(1000*lat.mean()),"median_latency_ms":float(1000*np.median(lat)),
        }
    kt=np.array([r["task_stop"]["k"] for r in records]); ke=np.array([r["estimator_stop"]["k"] for r in records])
    out["paired_task_minus_estimator_k"]=bootstrap_diff(kt,ke,991+sum(map(ord,task)))
    out["task_before_estimator_rate"]=float(np.mean(kt<ke))
    out["paired_completion_difference_task_minus_estimator"]=float(np.mean([int(r["task_stop"]["success"])-int(r["estimator_stop"]["success"]) for r in records]))
    return out


def evaluate_dataset(episodes, thresholds):
    tasks={}; detail={}
    for task in TASKS:
        records=[]
        for idx,ep in enumerate(episodes):
            row={"episode":idx}
            for p in ("fixed8","full","estimator_stop","task_stop"):
                row[p]=eval_policy(ep,task,p,thresholds[task].get("threshold"))
            records.append(row)
        tasks[task]=summarize(records,task); detail[task]=records
    return tasks, detail


def render_md(payload, path):
    lines=[
        "# Numerical ICP experiment", "",
        "> **[NumericalBackend]** Generated 3-D point clouds + iterative ICP/Kabsch were executed. No camera, neural pose model, contact physics, or physical robot was executed.", "",
        "The task gate uses an observable local dispersion proxy. A threshold is tuned on one split, frozen, checked on a disjoint safety-calibration split, then evaluated on held-out test seeds. Ground-truth pose error is never passed to the gate.", "",
        "## Frozen task thresholds", "",
        "| Task | threshold | safety-cal act | safety-cal unsafe ACT | unsafe 95% CI |", "|---|---:|---:|---:|---:|"
    ]
    for task,c in payload["calibration"]["tasks"].items():
        v=c["safety_check"]; ci=v["unsafe_act_wilson95"]
        lines.append(f"| {task} | {c['threshold']:.6g} | {100*v['act_rate']:.2f}% | {100*v['unsafe_act_rate']:.2f}% | [{100*ci[0]:.2f}%, {100*ci[1]:.2f}%] |")
    for label in ("in_distribution","stress_2x_sensor_noise"):
        lines += ["",f"## Held-out: {label}",""]
        for task,s in payload["evaluation"][label]["tasks"].items():
            lines += [f"### {task}","","| Policy | mean k | completion | HOLD | unsafe ACT | mean latency ms |","|---|---:|---:|---:|---:|---:|"]
            for p in ("fixed8","full","estimator_stop","task_stop"):
                x=s[p]; lines.append(f"| {p} | {x['mean_k']:.3f} | {100*x['completion_rate']:.2f}% | {100*x['hold_rate']:.2f}% | {100*x['unsafe_act_rate']:.2f}% | {x['mean_latency_ms']:.3f} |")
            d=s["paired_task_minus_estimator_k"]
            lines += ["",f"Paired `k_task-k_estimator`: {d['mean']:.3f}, bootstrap 95% CI [{d['bootstrap95'][0]:.3f}, {d['bootstrap95'][1]:.3f}].  ",f"`k_task < k_estimator`: {100*s['task_before_estimator_rate']:.2f}%.  ",f"Completion difference (task - estimator): {100*s['paired_completion_difference_task_minus_estimator']:.2f} percentage points.",""]
    lines += ["## Evidence ceiling","","These results support only the behavior of the public numerical registration fixture under its declared generator and split protocol. Wall-clock time is machine-specific. RGB-D/GPU/robot speedup and physical manipulation non-inferiority remain **HOLD / open hypotheses**.",""]
    path.write_text("\n".join(lines),encoding="utf-8")


def run(profile, out_dir):
    if profile=="ci": cfg=dict(tune_n=20,safety_n=20,test_per_seed=8,stress_n=16,max_iter=14,points=64)
    elif profile=="quick": cfg=dict(tune_n=40,safety_n=40,test_per_seed=16,stress_n=24,max_iter=16,points=80)
    else: cfg=dict(tune_n=96,safety_n=96,test_per_seed=40,stress_n=60,max_iter=20,points=120)
    rng=np.random.default_rng(TUNE_SEED); tune=[generate_episode(rng,cfg['max_iter'],cfg['points']) for _ in range(cfg['tune_n'])]
    rng=np.random.default_rng(SAFETY_SEED); safe=[generate_episode(rng,cfg['max_iter'],cfg['points']) for _ in range(cfg['safety_n'])]
    thresholds={}; cal_tasks={}
    for task in TASKS:
        t=tune_threshold(tune,task); th=t.get("threshold")
        check=evaluate_frozen_threshold(safe,task,th) if th is not None else {"episodes":len(safe),"act_rate":0.0,"hold_rate":1.0,"completion_rate":0.0,"unsafe_act_rate":0.0,"unsafe_act_wilson95":wilson(0,len(safe)),"mean_k":float(cfg['max_iter'])}
        thresholds[task]=t; cal_tasks[task]={**t,"safety_check":check}
    test=[]
    for seed in TEST_SEEDS:
        rng=np.random.default_rng(seed); test += [generate_episode(rng,cfg['max_iter'],cfg['points']) for _ in range(cfg['test_per_seed'])]
    rng=np.random.default_rng(STRESS_SEED); stress=[generate_episode(rng,cfg['max_iter'],cfg['points'],noise_scale=2.0) for _ in range(cfg['stress_n'])]
    id_tasks,id_detail=evaluate_dataset(test,thresholds); stress_tasks,stress_detail=evaluate_dataset(stress,thresholds)
    payload={
        "schema_version":3,"label":"[NumericalBackend]","backend":"numpy_point_to_point_icp",
        "evidence_boundary":{"generated_point_clouds":True,"rigid_registration_executed":True,"gate_has_hidden_ground_truth_access":False,"camera_executed":False,"rgbd_sensor_executed":False,"neural_pose_model_executed":False,"rigid_body_contact_physics_executed":False,"physical_robot_executed":False},
        "environment":{"python":sys.version.split()[0],"numpy":np.__version__,"platform":platform.platform()},
        "configuration":{**cfg,"tune_seed":TUNE_SEED,"safety_seed":SAFETY_SEED,"test_seeds":list(TEST_SEEDS),"stress_seed":STRESS_SEED},
        "calibration":{"method":"task-specific threshold tuning on observable local proxy; frozen threshold independently safety-checked","tasks":cal_tasks},
        "evaluation":{"in_distribution":{"episodes":len(test),"tasks":id_tasks},"stress_2x_sensor_noise":{"episodes":len(stress),"tasks":stress_tasks}},
        "detail":{"in_distribution":id_detail,"stress_2x_sensor_noise":stress_detail},
    }
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"numerical_icp.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    with (out_dir/"numerical_icp.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["condition","task","policy","mean_k","completion_rate","hold_rate","unsafe_act_rate","mean_latency_ms"])
        for condition in ("in_distribution","stress_2x_sensor_noise"):
            for task,s in payload["evaluation"][condition]["tasks"].items():
                for p in ("fixed8","full","estimator_stop","task_stop"):
                    x=s[p]; w.writerow([condition,task,p,x['mean_k'],x['completion_rate'],x['hold_rate'],x['unsafe_act_rate'],x['mean_latency_ms']])
    render_md(payload,out_dir/"NUMERICAL_RESULTS.md")
    return payload


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--profile",choices=("ci","quick","standard"),default="standard"); ap.add_argument("--out-dir",default="results"); args=ap.parse_args()
    p=run(args.profile,args.out_dir)
    concise={"calibration":p["calibration"],"evaluation":{c:{"episodes":v["episodes"],"tasks":v["tasks"]} for c,v in p["evaluation"].items()}}
    print(json.dumps(concise,indent=2))

if __name__=="__main__": main()
