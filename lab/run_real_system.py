#!/usr/bin/env python3
"""Backend-agnostic lab harness for coverage-qualified task stopping.

Input files are JSONL episodes. Online gate sees ONLY `stages[*].features`.
Oracle pose errors are used only in TRAIN/CALIBRATION/TEST evaluation.

Units:
  pose error = [tx, ty, tz, rx, ry, rz]
  translation in metres, rotation in radians.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Model:
    beta: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    error_floor: np.ndarray

def load_jsonl(path):
    rows=[]
    with Path(path).open("r", encoding="utf-8") as f:
        for ln,line in enumerate(f,1):
            if not line.strip(): continue
            try: r=json.loads(line)
            except Exception as e: raise ValueError(f"{path}:{ln}: invalid JSON: {e}")
            rows.append(r)
    if not rows: raise ValueError(f"{path}: no episodes")
    return rows

def validate_episode(ep, feature_dim=None):
    if not isinstance(ep.get("episode_id"), str) or not ep["episode_id"]:
        raise ValueError("episode_id must be a non-empty string")
    stages=ep.get("stages")
    errors=ep.get("oracle",{}).get("abs_pose_error_6d")
    if not isinstance(stages,list) or not stages: raise ValueError(f"{ep['episode_id']}: stages missing")
    if not isinstance(errors,list) or len(errors)!=len(stages):
        raise ValueError(f"{ep['episode_id']}: oracle.abs_pose_error_6d must match stages")
    ks=[]; stops=0
    for j,(s,e) in enumerate(zip(stages,errors)):
        k=s.get("k"); feat=s.get("features")
        if not isinstance(k,int): raise ValueError(f"{ep['episode_id']}: stage {j} k must be int")
        ks.append(k)
        if not isinstance(feat,list) or not feat: raise ValueError(f"{ep['episode_id']}: stage {j} features missing")
        if feature_dim is not None and len(feat)!=feature_dim:
            raise ValueError(f"{ep['episode_id']}: feature dim mismatch")
        if not all(math.isfinite(float(x)) for x in feat):
            raise ValueError(f"{ep['episode_id']}: non-finite feature")
        if not isinstance(e,list) or len(e)!=6 or any(float(x)<0 or not math.isfinite(float(x)) for x in e):
            raise ValueError(f"{ep['episode_id']}: oracle error must be six finite nonnegative values")
        inc=float(s.get("incremental_ms",0.0))
        if inc < 0 or not math.isfinite(inc): raise ValueError(f"{ep['episode_id']}: invalid incremental_ms")
        stops += bool(s.get("estimator_stop",False))
    if ks != sorted(ks) or len(set(ks)) != len(ks):
        raise ValueError(f"{ep['episode_id']}: stage k must be unique and sorted")
    if stops > 1: raise ValueError(f"{ep['episode_id']}: at most one estimator_stop=true")
    return len(stages[0]["features"])

def validate_dataset(rows):
    d=None; ids=set()
    for ep in rows:
        if ep.get("episode_id") in ids: raise ValueError(f"duplicate episode_id {ep.get('episode_id')}")
        ids.add(ep.get("episode_id")); d=validate_episode(ep,d)
    return d,ids

def ensure_disjoint(*datasets):
    seen=set()
    for name,rows in datasets:
        _,ids=validate_dataset(rows)
        dup=seen & ids
        if dup: raise ValueError(f"split leakage: {name} shares episode ids: {sorted(dup)[:5]}")
        seen |= ids

def design(feat, mean, scale):
    x=(np.asarray(feat,float)-mean)/scale
    return np.r_[1.0,x]

def fit_model(train, error_floor):
    feats=[np.asarray(s["features"],float) for ep in train for s in ep["stages"]]
    X0=np.asarray(feats,float); mean=X0.mean(0); scale=X0.std(0)
    scale=np.where(scale<1e-12,1.0,scale)
    X=np.vstack([design(f,mean,scale) for f in feats])
    floors=np.asarray(error_floor,float); betas=[]
    for axis in range(6):
        y=[]
        for ep in train:
            for err in ep["oracle"]["abs_pose_error_6d"]:
                y.append(math.log(float(err[axis])+float(floors[axis])))
        beta,*_=np.linalg.lstsq(X,np.asarray(y,float),rcond=None); betas.append(beta)
    return Model(np.asarray(betas,float),mean,scale,floors)

def predicted_log_error(stage, axis, model):
    return float(design(stage["features"],model.mean,model.scale) @ model.beta[axis])

def nonconformity(ep, model):
    worst=-math.inf
    for s,err in zip(ep["stages"],ep["oracle"]["abs_pose_error_6d"]):
        for axis in range(6):
            val=math.log(float(err[axis])+float(model.error_floor[axis]))-predicted_log_error(s,axis,model)
            worst=max(worst,val)
    return float(worst)

def conformal_quantile(scores, alpha):
    x=np.sort(np.asarray(scores,float)); n=len(x)
    rank=int(math.ceil((n+1)*(1-alpha))); rank=min(max(rank,1),n)
    return float(x[rank-1]),rank

def completion_bounds(stage, model, q):
    out=[]
    for axis in range(6):
        upper=math.exp(predicted_log_error(stage,axis,model)+float(q))-float(model.error_floor[axis])
        out.append(max(0.0,upper))
    return np.asarray(out,float)

def task_pass(spec, x):
    a=np.abs(np.asarray(x,float)); mask=[int(i) for i in spec["mask"]]; tol=np.asarray(spec["tol"],float)
    if spec["kind"]=="box": return all(a[i] <= tol[i] for i in mask)
    if spec["kind"]=="l1": return sum(a[i]/tol[i] for i in mask) <= 1.0
    raise ValueError(f"unsupported task kind {spec['kind']}")

def first_certificate_stage(stages, spec, model, q):
    for idx,s in enumerate(stages):
        if task_pass(spec, completion_bounds(s,model,q)): return idx,True
    return len(stages)-1,False

def estimator_index(stages):
    for i,s in enumerate(stages):
        if bool(s.get("estimator_stop",False)): return i
    return len(stages)-1

def cumulative_ms(stages, idx):
    return float(sum(float(s.get("incremental_ms",0.0)) for s in stages[:idx+1]))

def wilson(k,n,z=1.959963984540054):
    if n==0:return [None,None]
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.0,c-h),min(1.0,c+h)]

def boot_mean_diff(a,b,seed=12345,repeats=5000):
    a=np.asarray(a,float);b=np.asarray(b,float);d=a-b;n=len(d);rng=np.random.default_rng(seed);vals=np.empty(repeats)
    for i in range(repeats):
        idx=rng.integers(0,n,size=n); vals[i]=d[idx].mean()
    return {"mean":float(d.mean()),"bootstrap95":[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}

def evaluate(test, tasks, model, q, margin, repeats):
    covered=[nonconformity(ep,model)<=q+1e-12 for ep in test]; cov_n=sum(covered);n=len(test)
    result={"episodes":n,"trajectory_envelope_coverage":cov_n/n,"trajectory_envelope_coverage_wilson95":wilson(cov_n,n),"tasks":{}}
    for ti,(name,spec) in enumerate(tasks.items()):
        kc=[];ke=[];c_succ=[];e_succ=[];hold=[];unsafe=[];c_ms=[];e_ms=[];unsafe_cov=0
        for ep,cov in zip(test,covered):
            ci,act=first_certificate_stage(ep["stages"],spec,model,q); ei=estimator_index(ep["stages"])
            cerr=np.asarray(ep["oracle"]["abs_pose_error_6d"][ci],float); eerr=np.asarray(ep["oracle"]["abs_pose_error_6d"][ei],float)
            cs=bool(act and task_pass(spec,cerr)); es=bool(task_pass(spec,eerr)); un=bool(act and not cs)
            if un and cov: unsafe_cov += 1
            kc.append(ep["stages"][ci]["k"]);ke.append(ep["stages"][ei]["k"])
            c_succ.append(int(cs));e_succ.append(int(es));hold.append(int(not act));unsafe.append(int(un))
            c_ms.append(cumulative_ms(ep["stages"],ci));e_ms.append(cumulative_ms(ep["stages"],ei))
        kd=boot_mean_diff(kc,ke,1000+ti,repeats); cd=boot_mean_diff(c_succ,e_succ,2000+ti,repeats); ld=boot_mean_diff(c_ms,e_ms,3000+ti,repeats)
        result["tasks"][name]={
          "completion_stop":{"mean_k":float(np.mean(kc)),"completion_rate":float(np.mean(c_succ)),"hold_rate":float(np.mean(hold)),"unsafe_act_rate":float(np.mean(unsafe)),"mean_perception_ms":float(np.mean(c_ms))},
          "estimator_stop":{"mean_k":float(np.mean(ke)),"completion_rate":float(np.mean(e_succ)),"mean_perception_ms":float(np.mean(e_ms))},
          "completion_before_estimator_rate":float(np.mean(np.asarray(kc)<np.asarray(ke))),
          "paired_k_completion_minus_estimator":kd,"paired_completion_difference":cd,"paired_perception_ms_difference":ld,
          "noninferiority_margin":float(margin),"noninferiority_pass":bool(cd["bootstrap95"][0] >= -float(margin)),
          "earlier_stage_pass":bool(kd["bootstrap95"][1] < 0.0),"latency_reduction_pass":bool(ld["bootstrap95"][1] < 0.0),
          "unsafe_act_on_covered_episode_count":int(unsafe_cov)}
        if unsafe_cov != 0: raise AssertionError(f"{name}: unsafe ACT occurred on an episode covered by the calibrated envelope")
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--train",required=True); ap.add_argument("--calibration",required=True); ap.add_argument("--test",required=True); ap.add_argument("--out",default="lab_results.json"); args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8")); train=load_jsonl(args.train);cal=load_jsonl(args.calibration);test=load_jsonl(args.test)
    ensure_disjoint(("train",train),("calibration",cal),("test",test)); d1,_=validate_dataset(train);d2,_=validate_dataset(cal);d3,_=validate_dataset(test)
    if len({d1,d2,d3})!=1: raise ValueError("feature dimension differs across splits")
    alpha=float(cfg.get("alpha",0.1)); floors=cfg.get("error_floor",[1e-5,1e-5,1e-5,1e-4,1e-4,1e-4])
    model=fit_model(train,floors); scores=[nonconformity(ep,model) for ep in cal]; q,rank=conformal_quantile(scores,alpha)
    ev=evaluate(test,cfg["tasks"],model,q,float(cfg.get("noninferiority_margin",0.05)),int(cfg.get("bootstrap_repeats",5000)))
    payload={"schema_version":1,"claim":"coverage-qualified downstream task stopping of iterative 6D pose refinement","units":{"translation":"metres","rotation":"radians","time":"milliseconds"},"protocol":{"train_episodes":len(train),"calibration_episodes":len(cal),"test_episodes":len(test),"alpha":alpha,"target_marginal_whole_trajectory_coverage":1-alpha,"calibration_quantile_rank":rank,"q":q,"feature_dim":d1},"evaluation":ev,"evidence_boundary":{"online_gate_uses_oracle":False,"physical_robot_result_inferred_from_pose_error":False,"result_scope":"instrumented iterative 6D pose backend; task readers are declared pose-error admissibility tests"}}
    Path(args.out).write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
