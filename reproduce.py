#!/usr/bin/env python3
"""One-command reproduction entry point."""
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--profile',choices=('ci','quick','standard'),default='quick'); args=ap.parse_args()
    run([sys.executable,'-m','unittest','discover','-s','tests','-v'])
    run([sys.executable,'evidence/validate_claim_card.py'])
    run([sys.executable,'theory/completion_envelope.py'])
    run([sys.executable,'benchmark.py'])
    run([sys.executable,'experiments/numerical_icp.py','--profile',args.profile,'--out-dir','artifacts/numerical'])
    run([sys.executable,'experiments/conformal_completion.py','--profile',args.profile,'--out-dir','artifacts/conformal-raw'])
    run([sys.executable,'experiments/learned_conformal_completion.py','--profile',args.profile,'--out-dir','artifacts/conformal-learned'])
    run([sys.executable,'experiments/replication_study.py','--profile',args.profile,'--out-dir','artifacts/replication'])
    print('\nReproduction complete.')
    print('Numerical threshold results: artifacts/numerical/NUMERICAL_RESULTS.md')
    print('Raw-proxy conformal ablation: artifacts/conformal-raw/CONFORMAL_COMPLETION_RESULTS.md')
    print('Learned-shape calibrated certificate: artifacts/conformal-learned/LEARNED_CONFORMAL_RESULTS.md')
    print('Repeated calibration-test study: artifacts/replication/REPLICATION_RESULTS.md')
    print('Theory/status: theory/TOLEDO_COMPLETION_ENVELOPE.md and theory/COVERAGE_TO_TASK_CERTIFICATE.md')

if __name__=='__main__': main()
