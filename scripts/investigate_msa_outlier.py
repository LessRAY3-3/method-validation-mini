"""Prespecified targeted CPU follow-up; default prepares only, --execute runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from run_prediction_ablation import parse_a3m

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/prediction_msa_diagnosis/followup'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(base):
    if (base / 'plan.json').exists():
        return json.loads((base / 'plan.json').read_text())
    if base.exists():
        raise ValueError('Existing directory without plan; choose a new directory')
    old = ROOT / 'results/prediction_ablation'
    original = json.loads((old / 'plan.json').read_text())
    inputs = {c: (old / original['inputs'][c]['path']).read_text() for c in original['conditions_in_execution_order']}
    records = parse_a3m(inputs['deduplicated_saved_msa'], original['sequence'])
    q, h = records
    def encode(rows):
        return '#20\t1\n' + ''.join(f'>row_{i}\n{seq}\n' for i, (_, seq) in enumerate(rows))
    inputs['duplicate_query'] = encode([q, q, h])
    inputs['duplicate_homolog'] = encode([q, h, h])
    (base / 'inputs').mkdir(parents=True)
    info = {}
    for name, text in inputs.items():
        path = base / 'inputs' / f'{name}.a3m';path.write_text(text)
        info[name] = {'path': str(path.relative_to(base)), 'sha256': sha(path)}
    specs = [{'id':'repeat_bad_seed','condition':'deduplicated_saved_msa','seed':20261110,'recycles':3,'extra_args':[]}]
    for name in ('duplicate_query','duplicate_homolog'):
        specs.append({'id':name+'_bad_seed','condition':name,'seed':20261110,'recycles':3,'extra_args':[]})
    specs.append({'id':'raw_capacity_2_bad_seed','condition':'raw_saved_msa','seed':20261110,'recycles':3,'extra_args':['--max-msa','2:1']})
    specs.append({'id':'dedup_12_recycles_bad_seed','condition':'deduplicated_saved_msa','seed':20261110,'recycles':12,'extra_args':['--recycle-early-stop-tolerance','0']})
    for seed in (20261111,20261112,20261113):
        for condition in ('query_only','deduplicated_saved_msa','raw_saved_msa'):
            specs.append({'id':f'extension_{condition}_{seed}','condition':condition,'seed':seed,'recycles':3,'extra_args':[]})
    plan = {'created_before_followup_utc':datetime.now(timezone.utc).isoformat(),
            'reason':'Investigate a post-hoc outlier, not a confirmatory population study',
            'baseline_plan_sha256':sha(old/'plan.json'),'sequence':original['sequence'],
            'weights_sha256':original['weights_model_1_ptm_sha256'],'inputs':info,'runs':specs,
            'questions':['Is the original outlier exactly reproducible?','Do duplicate query versus duplicate homolog rows change the processed input/result?','Does lowering raw MSA cluster capacity reproduce the anomaly?','Does 12-recycle refinement change this same outlier?','Does paired extension to six seeds show the outlier in other seeds?'],
            'limits':'Only three new paired seeds; 38 NMR models are not independent replicates. The 12-recycle perturbation is outlier-specific, not evidence for universally optimal settings. No omitted seeds or replacement of historical results.',
            'device':'CPU','model':'alphafold2_ptm_model_1','templates':False,'relax':False,
            'analysis':'Median global CA RMSD over all 38 NMR models plus local structure; preserve all runs, including poor results.'}
    (base/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output-dir',type=Path,default=BASE)
    cli.add_argument('--execute',action='store_true')
    cli.add_argument('--weights-dir',type=Path,default=Path.home()/'Library/Caches/colabfold')
    args=cli.parse_args();base=args.output_dir.resolve();plan=prepare(base)
    if not args.execute:
        print(f'Prepared {len(plan["runs"])} short CPU runs; no inference launched: {base}');return
    weights=args.weights_dir/'params/params_model_1_ptm.npz'
    if sha(weights)!=plan['weights_sha256'] or not (args.weights_dir/'params/download_finished.txt').exists():
        raise ValueError('Weights do not match original experiment')
    env=dict(os.environ,JAX_PLATFORMS='cpu',JAX_COMPILATION_CACHE_DIR=str(ROOT/'.cache/colabfold_jax'),MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    for spec in plan['runs']:
        if (base/'STOP').exists():
            print('STOP present; no further runs launched');return
        folder=base/'runs'/spec['id'];manifest=folder/'manifest.json'
        if manifest.exists():
            record=json.loads(manifest.read_text())
            current={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'}
            if current!=record['outputs_sha256'] or record['plan_sha256']!=sha(base/'plan.json'):
                raise ValueError('Completed-run evidence changed')
            continue
        if folder.exists():raise ValueError(f'Incomplete directory preserved, refusing overwrite: {folder}')
        msa=base/plan['inputs'][spec['condition']]['path']
        if sha(msa)!=plan['inputs'][spec['condition']]['sha256']:raise ValueError('Input changed')
        folder.mkdir(parents=True)
        cmd=[str(ROOT/'.colabfold/bin/python'),'-m','colabfold.batch',str(msa),str(folder),'--data',str(args.weights_dir),'--model-type','alphafold2_ptm','--model-order','1','--num-models','1','--num-seeds','1','--random-seed',str(spec['seed']),'--num-recycle',str(spec['recycles']),'--rank','plddt','--compile-mode','fast',*spec['extra_args']]
        started=datetime.now(timezone.utc).isoformat()
        print('Running '+spec['id'],flush=True)
        with (folder/'run.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        record={'spec':spec,'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'command':cmd,'plan_sha256':sha(base/'plan.json'),'input_sha256':sha(msa),'weights_sha256':sha(weights),'outputs_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}
        manifest.write_text(json.dumps(record,indent=2)+'\n')
    print('All follow-up runs completed; no background work')


if __name__=='__main__':main()
