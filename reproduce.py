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
    run([sys.executable,'benchmark.py'])
    run([sys.executable,'experiments/numerical_icp.py','--profile',args.profile,'--out-dir','artifacts/numerical'])
    print('\nReproduction complete. See artifacts/numerical/NUMERICAL_RESULTS.md')

if __name__=='__main__': main()
