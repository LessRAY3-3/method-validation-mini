"""Wait for the frozen GPU queue, then compare once; never launches dynamics."""
import argparse, fcntl, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison-sha256',required=True)
    p.add_argument('--timeout-seconds',type=int,default=14400)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    setup=root/'setup/confirmation';setup.mkdir(parents=True,exist_ok=True)
    lock=(setup/'postlude.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'waiting','pid':os.getpid(),'started_unix':time.time(),'comparison_sha256':a.comparison_sha256}
    destination=root/'results/md_confirmation_comparison'
    if destination.exists():raise ValueError('Comparison destination exists; refusing automatic overwrite/re-execution')
    def save():
        state['updated_unix']=time.time();temp=setup/'postlude_state.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(setup/'postlude_state.json')
    child=None
    def stop(signum,frame):
        if child is not None and child.poll() is None:child.terminate()
        state['status']='paused';save();raise SystemExit(1)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save()
    try:
        while True:
            queue=json.loads((root/'results/md_replicates_gpu_confirmation/queue_status.json').read_text())
            state['queue_status']=queue['status']
            if queue['status']=='complete':break
            if queue['status'] in ('failed','paused'):raise RuntimeError('Queue stopped: '+queue['status'])
            if time.time()-state['started_unix']>a.timeout_seconds:raise TimeoutError('Queue wait exceeded declared timeout')
            os.kill(queue['queue_pid'],0)
            save();time.sleep(15)
        script=root/'scripts/compare_md_confirmation_batches.py'
        if hashlib.sha256(script.read_bytes()).hexdigest()!=a.comparison_sha256:raise ValueError('Comparison source changed after postlude launch')
        cmd=[sys.executable,str(script),'--new-root',str(root/'results/md_replicates_gpu_confirmation'),'--old-analysis',str(root/'results/md_confirmation_preparation/original_batch_same_analysis'),'--old-series-root',str(setup/'old_audited_timeseries'),'--output-dir',str(destination)]
        state.update(status='comparing',argv=cmd);save()
        child=subprocess.Popen(cmd,cwd=root);code=child.wait();child=None
        state['exit_code']=code
        if code:raise RuntimeError('Comparison failed, inspect postlude.log')
        state.update(status='complete',completed_unix=time.time());save()
        return 0
    except Exception as error:
        state.update(status='failed',error=str(error));save();raise

if __name__=='__main__':raise SystemExit(main())
