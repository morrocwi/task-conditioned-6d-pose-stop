#!/usr/bin/env python3
"""Generate software-only smoke-test data for the lab harness.

These records are NOT research evidence.
"""
import argparse, json, math
from pathlib import Path
import numpy as np

def make_episode(rng,eid,max_iter=12):
    e0=np.r_[rng.uniform(.004,.012,3), rng.uniform(math.radians(2),math.radians(12),3)]
    stages=[]; errors=[]; est_stop=max_iter-2
    for k in range(max_iter+1):
        decay=0.72**k
        err=np.abs(e0*decay + rng.normal(0,[.00015]*3+[math.radians(.08)]*3))
        proxy=np.abs(err*(1+rng.normal(0,.18,6))) + np.r_[np.repeat(.00005,3),np.repeat(math.radians(.03),3)]
        delta=np.abs(e0*(0.72**k)*(1-.72)); rms=float(np.mean(proxy[:3]))
        feat=[k/max_iter, math.log(rms+1e-6)] + [math.log(float(x)+1e-7) for x in proxy] + [math.log(float(x)+1e-7) for x in delta]
        stages.append({"k":k,"features":feat,"incremental_ms":2.0+0.1*k,"estimator_stop":k==est_stop})
        errors.append([float(x) for x in err])
    return {"episode_id":eid,"stages":stages,"oracle":{"abs_pose_error_6d":errors}}

def write(path,seed,n,prefix):
    rng=np.random.default_rng(seed)
    with Path(path).open("w",encoding="utf-8") as f:
        for i in range(n): f.write(json.dumps(make_episode(rng,f"{prefix}-{i:04d}"))+"\n")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-dir",required=True); ap.add_argument("--n",type=int,default=40); a=ap.parse_args()
    d=Path(a.out_dir); d.mkdir(parents=True,exist_ok=True)
    write(d/"train.jsonl",11,a.n,"train"); write(d/"calibration.jsonl",22,a.n,"cal"); write(d/"test.jsonl",33,a.n,"test")
if __name__=="__main__": main()
