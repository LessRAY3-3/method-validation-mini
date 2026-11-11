#!/usr/bin/env python3
"""Sequential GPU batches, with frozen input/source checks for confirmation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

STUDIES = {
    'original': ((20261110, 20261111, 20261112), 'results/md_replicates_gpu'),
    'confirmation': ((20261113, 20261114, 20261115), 'results/md_replicates_gpu_confirmation'),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_protocol(root, path, seeds, output_relative):
    p=json.loads(path.read_text())
    if p['seeds'] != list(seeds) or p['output_root'] != output_relative or p['production_ns_each'] != 11:
        raise ValueError('Protocol does not match this fixed batch')
    if p.get('nvt_ps') != 100 or p.get('npt_ps') != 100:
        raise ValueError('Equilibration protocol changed')
    for field in ('input_sha256', 'source_sha256'):
        if not p.get(field):raise ValueError('Missing frozen '+field)
        for name,want in p[field].items():
            if sha(root/name)!=want:raise ValueError('Frozen file changed: '+name)
    return p

def require_gpu_logs(directory):
    for stage in ('nvt','npt','md'):
        s=(directory/f'{stage}.log').read_text()
        if 'short-ranged interactions on the GPU' not in s or 'PME tasks will do all aspects on the GPU' not in s:
            raise ValueError(f'GPU offload not confirmed for {directory.name}/{stage}')
        if 'LINCS WARNING' in s or 'Fatal error' in s:
            raise ValueError(f'Numerical warning/error in {directory.name}/{stage}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gmx',required=True)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--gpu-id',default='0')
    parser.add_argument('--study',choices=STUDIES,default='original')
    parser.add_argument('--protocol',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    seeds,relative=STUDIES[args.study];output=root/relative
    if args.study=='confirmation' and args.protocol is None:parser.error('Confirmation requires --protocol')
    if args.threads!=4 and args.study=='confirmation':parser.error('Frozen confirmation uses four threads')
    protocol=None
    if args.protocol:
        args.protocol=args.protocol.resolve()
        protocol=validate_protocol(root,args.protocol,seeds,relative)
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if protocol:
        snapshot=output/'protocol_snapshot.json'
        if snapshot.exists() and sha(snapshot)!=sha(args.protocol):raise ValueError('Existing batch protocol differs')
        if not snapshot.exists():snapshot.write_bytes(args.protocol.read_bytes())
    state=dict(status='running',study=args.study,queue_pid=os.getpid(),seeds=list(seeds),production_ns_each=11,completed_seeds=[],start_unix=time.time(),protocol_sha256=sha(args.protocol) if args.protocol else None)
    def check_frozen():
        if protocol:
            if sha(args.protocol)!=state['protocol_sha256']:
                raise ValueError('Protocol changed after queue launch')
            validate_protocol(root,args.protocol,seeds,relative)
    def save():
        state['updated_unix']=time.time();tmp=output/'queue_status.tmp'
        tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(output/'queue_status.json')
    child=None;stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(cmd):
        nonlocal child
        print('Starting: '+repr(cmd),flush=True)
        child=subprocess.Popen(cmd,cwd=root);code=child.wait();child=None
        if code or stopping:
            state.update(status='paused' if stopping else 'failed',exit_code=code);save()
            return False
        return True
    try:
        for seed in seeds:
            if stopping:break
            check_frozen()
            state.update(active_seed=seed,phase='dynamics_and_per_run_analysis');save()
            cmd=[sys.executable,'-u',str(root/'scripts/run_md_replicate.py'),'--seed',str(seed),'--total-ns','11','--threads',str(args.threads),'--gmx',args.gmx,'--execution','cuda','--gpu-id',args.gpu_id,'--output-root',relative]
            if not run(cmd):return 1
            require_gpu_logs(output/f'seed_{seed}')
            state['completed_seeds'].append(seed);save()
        if not stopping:
            check_frozen()
            state.update(status='analyzing',phase='batch_analysis');save()
            script='analyze_md_confirmation_batch.py' if args.study=='confirmation' else 'analyze_md_replicates.py'
            destination=output/('confirmation_analysis' if args.study=='confirmation' else 'aggregate_analysis')
            # Existing output is deliberately not overwritten. An interrupted analysis
            # must be inspected separately instead of silently replacing evidence.
            cmd=[sys.executable,str(root/'scripts'/script),'--input-root',str(output),'--gmx',args.gmx,'--output-dir',str(destination)]
            if args.study=='confirmation':cmd+=['--seeds',*[str(s) for s in seeds]]
            else:cmd+=['--total-ns','11']
            if not run(cmd):return 1
        state['status']='paused' if stopping else 'complete';save()
        return 0 if not stopping else 1
    except Exception as error:
        state.update(status='failed',error=str(error));save();raise

if __name__=='__main__':raise SystemExit(main())
