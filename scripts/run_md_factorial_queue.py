#!/usr/bin/env python3
"""Execute all nine prospectively fixed factorial branches, then analyze once."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--gmx',required=True)
    a=p.parse_args();protocol=a.protocol.resolve();frozen=sha(protocol);spec=json.loads(protocol.read_text())
    out=ROOT/spec['output_root'];out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot=out/'protocol_snapshot.json'
    if snapshot.exists() and sha(snapshot)!=frozen:raise ValueError('Existing protocol differs')
    if not snapshot.exists():snapshot.write_bytes(protocol.read_bytes())
    state=dict(status='running',pid=os.getpid(),started_unix=time.time(),protocol_sha256=frozen,completed=[])
    child=None;stopping=False
    def save():
        state['updated_unix']=time.time();tmp=out/'queue_status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'queue_status.json')
    def verify():
        if sha(protocol)!=frozen:raise ValueError('Protocol changed')
        for field in ('source_sha256','input_sha256'):
            for rel,want in spec[field].items():
                if sha(ROOT/rel)!=want:raise ValueError('Frozen file changed: '+rel)
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(cmd):
        nonlocal child
        if stopping:raise InterruptedError('Stop requested')
        print(repr(cmd),flush=True);child=subprocess.Popen(cmd,cwd=ROOT);code=child.wait();child=None
        if stopping:raise InterruptedError('Stop requested')
        if code:raise RuntimeError(f'Child exited {code}')
    save()
    try:
        for cell in ('B','C','D'):
            for seed in spec['seeds']:
                verify();state.update(cell=cell,seed=seed,phase='dynamics');save()
                run([sys.executable,'-u',str(ROOT/'scripts/run_md_factorial.py'),'--protocol',str(protocol),'--gmx',a.gmx,'--cell',cell,'--seed',str(seed)])
                state['completed'].append(f'{cell}/{seed}');save()
        verify();state.update(status='analyzing',phase='factorial_analysis');save()
        run([sys.executable,'-u',str(ROOT/'scripts/analyze_md_factorial.py'),'--gmx',a.gmx,'--output-dir',str(out/'analysis')])
        verify();state.update(status='complete',completed_unix=time.time());save()
    except BaseException as e:
        state.update(status='paused' if stopping else 'failed',error=str(e));save();raise
if __name__=='__main__':main()
