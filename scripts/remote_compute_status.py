"""Read-only SSH status for the two authorized experiment hosts; no job launches."""
import argparse
from datetime import datetime, timezone
import json
import subprocess

HOSTS = {
    'm3': '/Users/leisirui/codex-runs/method-validation-20261110-m3',
    'win4070': '/home/lessray/codex-runs/method-validation-20261110-gpu',
}
REMOTE = r'''
from pathlib import Path
import json,os,sys,re
root=Path(sys.argv[1]); result={'root':str(root),'exists':root.exists()}
def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None
def tail(path,n=5):
    try:return path.read_text(errors='replace').splitlines()[-n:]
    except OSError:return []
def alive(pid):
    try:os.kill(int(pid),0);return True
    except (OSError,ValueError,TypeError):return False
def batch_status(base,seeds,analysis_name):
    result={'exists':base.exists(),'queue':read_json(base/'queue_status.json'),'seeds':{}}
    if result['queue']:result['queue_process_alive']=alive(result['queue'].get('queue_pid'))
    for seed in seeds:
        d=base/f'seed_{seed}'
        stage={'completed_stages':[s for s in ('nvt','npt','md') if (d/f'{s}.done.json').exists()]}
        try:
            log=(d/'md.log').read_text(errors='replace')
            steps=re.findall(r'Step\s+Time\s*\n\s*(\d+)\s+([\d.]+)',log)
            if steps:stage.update(last_logged_step=int(steps[-1][0]),production_ns=float(steps[-1][1])/1000)
            stage['lincs_warnings']=log.count('LINCS WARNING')
            stage['fatal_errors']=log.count('Fatal error')
        except OSError:pass
        result['seeds'][str(seed)]=stage
    summary=read_json(base/analysis_name/'summary.json')
    result['aggregate_status']={k:summary.get(k) for k in ('status','n_trajectories')} if summary else None
    return result
if sys.argv[2]=='m3':
    base=root/'results/docking_benchmark_m3'; counts={}
    for e in (1,8,32):
        for seed in (42,2026,20261110,20261111,20261112):
            run=read_json(base/f'e{e}_s{seed}'/'run.json')
            state=run.get('status','unknown') if run else 'pending'
            counts[state]=counts.get(state,0)+1
    result['run_states']=counts
    result['runner_log_tail']=tail(base/'runner.log')
    result['exit_code_file']=tail(base/'runner_exit_code.txt',1)
else:
    setup=root/'setup'; base=root/'results/md_replicates_gpu'
    result['environment_ready']=(root/'env/bin/gmx').is_file()
    result['installation_log_tail']=tail(setup/'install.log')
    result['queue']=read_json(base/'queue_status.json')
    if result['queue']:
        result['queue_process_alive']=alive(result['queue'].get('queue_pid'))
    result['seeds']={}
    for seed in (20261110,20261111,20261112):
        d=base/f'seed_{seed}'
        result['seeds'][str(seed)]={'completed_stages':[s for s in ('nvt','npt','md') if (d/f'{s}.done.json').exists()], 'workflow_tail':tail(d/'workflow.log',3)}
    result['aggregate_status']=read_json(base/'aggregate_analysis/summary.json')
    if result['aggregate_status']:
        result['aggregate_status']={k:result['aggregate_status'].get(k) for k in ('status','n_trajectories')}
    result['confirmation']=batch_status(root/'results/md_replicates_gpu_confirmation',(20261113,20261114,20261115),'confirmation_analysis')
    comparison=read_json(root/'results/md_confirmation_comparison/summary.json')
    result['confirmation']['comparison_status']=comparison.get('status') if comparison else None
    result['confirmation']['postlude']=read_json(setup/'confirmation/postlude_state.json')
    factorial=root/'results/md_factorial'
    result['factorial']={'queue':read_json(factorial/'queue_status.json'),'cells':{cell:batch_status(factorial/cell,(20261113,20261114,20261115),'analysis') for cell in ('B','C','D')}}
    fs=read_json(factorial/'analysis/summary.json')
    result['factorial']['analysis']={key:fs.get(key) for key in ('status','n_cells','n_seed_branches_per_cell','frames_per_run')} if fs else None
print(json.dumps(result))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',choices=list(HOSTS),action='append')
    args=parser.parse_args()
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'hosts':{}}
    for host in args.host or list(HOSTS):
        try:
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3','-',HOSTS[host],host],input=REMOTE,text=True,capture_output=True,timeout=25)
            result['hosts'][host]=json.loads(p.stdout) if p.returncode==0 else {'error':p.stderr.strip(),'exit_code':p.returncode}
        except (subprocess.TimeoutExpired,ValueError) as e:
            result['hosts'][host]={'error':str(e)}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
