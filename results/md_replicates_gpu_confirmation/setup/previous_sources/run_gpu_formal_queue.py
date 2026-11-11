#!/usr/bin/env python3
"""Sequential formal GPU runs; stop on failure and resume with the same command."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gmx', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--gpu-id', default='0')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / 'results/md_replicates_gpu'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='running', queue_pid=os.getpid(), seeds=[20261110, 20261111, 20261112], production_ns_each=11, completed_seeds=[], start_unix=time.time())
    def save():
        state['updated_unix'] = time.time()
        tmp = output / 'queue_status.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(output / 'queue_status.json')
    child = None
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for seed in state['seeds']:
        if stopping:
            break
        state['active_seed'] = seed
        save()
        cmd = [sys.executable, '-u', str(root/'scripts/run_md_replicate.py'), '--seed', str(seed), '--total-ns', '11', '--threads', str(args.threads), '--gmx', args.gmx, '--execution', 'cuda', '--gpu-id', args.gpu_id, '--output-root', 'results/md_replicates_gpu']
        print(f'Starting seed {seed}: {cmd}', flush=True)
        child = subprocess.Popen(cmd, cwd=root)
        code = child.wait()
        child = None
        if code or stopping:
            state.update(status='paused' if stopping else 'failed', exit_code=code)
            save()
            return 1
        state['completed_seeds'].append(seed)
        save()
    if not stopping:
        state['status'] = 'analyzing'
        save()
        cmd = [sys.executable, str(root/'scripts/analyze_md_replicates.py'), '--input-root', str(output), '--gmx', args.gmx, '--total-ns', '11', '--output-dir', str(output/'aggregate_analysis')]
        child = subprocess.Popen(cmd, cwd=root)
        code = child.wait()
        child = None
        if code or stopping:
            state.update(status='paused' if stopping else 'failed', phase='aggregate_analysis', exit_code=code)
            save()
            return 1
    state['status'] = 'paused' if stopping else 'complete'
    save()
    return 0 if not stopping else 1


if __name__ == '__main__':
    raise SystemExit(main())
