"""Post-hoc recycle coordinate trace. --execute required for one short CPU run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prediction_utils import ca_models, fitted_ca_rmsd
from prediction_provenance import sha256
from analyze_msa_followup import write_csv

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/prediction_msa_diagnosis/recycle_trace'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    baseline=ROOT/'results/prediction_msa_diagnosis/followup/runs/dedup_12_recycles_bad_seed'
    original=json.loads((baseline/'manifest.json').read_text())
    command=list(original['command'])
    command[4]=str(BASE/'run')
    command += ['--save-recycles']
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Locate structural change across recycling after observing the 12-recycle rescue; exploratory, not a new unbiased endpoint','baseline_manifest_sha256':sha256(baseline/'manifest.json'),'command':command,'sequence':'NLYIQWLKDGGPSSGRPPPS'}
    if not BASE.exists():
        BASE.mkdir(parents=True);(BASE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    else:
        stored=json.loads((BASE/'plan.json').read_text())
        if stored['command']!=command or stored['baseline_manifest_sha256']!=plan['baseline_manifest_sha256']:raise ValueError('Trace plan changed')
        plan=stored
    manifest=BASE/'manifest.json'
    if not manifest.exists():
        if not args.execute:
            print('Prepared one coordinate trace; no inference launched');return
        folder=BASE/'run'
        if folder.exists():raise ValueError('Incomplete run exists; refusing overwrite')
        folder.mkdir()
        env=dict(os.environ,JAX_PLATFORMS='cpu',JAX_COMPILATION_CACHE_DIR=str(ROOT/'.cache/colabfold_jax'),MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        with (folder/'run.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest.write_text(json.dumps({'plan_sha256':sha256(BASE/'plan.json'),'outputs':{p.name:sha256(p) for p in folder.iterdir() if p.is_file()}},indent=2)+'\n')
    evidence=json.loads(manifest.read_text());folder=BASE/'run'
    if evidence['plan_sha256']!=sha256(BASE/'plan.json') or evidence['outputs']!={p.name:sha256(p) for p in folder.iterdir() if p.is_file()}:raise ValueError('Trace evidence changed')
    nmr=ca_models(ROOT/'data/raw/1L2Y.pdb',plan['sequence'])
    paths=[]
    for p in folder.glob('*.pdb'):
        m=re.search(r'[._]r(\d+)\.pdb$',p.name)
        if m: paths.append((int(m[1]),p))
    paths.sort()
    if [i for i,p in paths]!=list(range(13)):raise ValueError(f'Missing recycle snapshots: {paths}')
    rows=[]
    for i,p in paths:
        xyz,confidence=ca_models(p,plan['sequence'])[0][1:]
        rmsds=[fitted_ca_rmsd(xyz,ref) for _,ref,_ in nmr]
        rows.append({'recycle':i,'mean_plddt_from_rounded_pdb':float(confidence.mean()),'median_ca_rmsd_angstrom':float(np.median(rmsds)),'model1_ca_rmsd_angstrom':rmsds[0]})
    reference=next(baseline.glob('*unrelaxed*.pdb'))
    final_equal=bool(np.array_equal(ca_models(paths[-1][1],plan['sequence'])[0][1],ca_models(reference,plan['sequence'])[0][1]))
    first3=ROOT/'results/prediction_ablation/runs/deduplicated_saved_msa/seed_20261110'
    at3_equal=bool(np.array_equal(ca_models(paths[3][1],plan['sequence'])[0][1],ca_models(next(first3.glob('*unrelaxed*.pdb')),plan['sequence'])[0][1]))
    if not final_equal or not at3_equal:raise ValueError('Trace did not preserve original stage3/final12 coordinates')
    write_csv(BASE/'per_recycle.csv',rows)
    (BASE/'summary.json').write_text(json.dumps({'rows':rows,'recycle3_matches_original_outlier':at3_equal,'recycle12_matches_prior_followup':final_equal,'interpretation':'Recycling steps are model refinement iterations, not physical time, MD frames, a folding pathway, or a general recommendation to select the best iteration.'},indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(8,6.4),sharex=True)
    axes[0].plot(range(13),[r['median_ca_rmsd_angstrom'] for r in rows],marker='o',color='#a44842')
    axes[0].set(ylabel='Median CA RMSD to NMR (Å)',title='Same input and seed: structure across model recycling')
    axes[1].plot(range(13),[r['mean_plddt_from_rounded_pdb'] for r in rows],marker='o',color='#287f8e')
    axes[1].set(xlabel='Model recycle index (not physical time)',ylabel='Mean pLDDT',xticks=range(13))
    for ax in axes:ax.axvline(3,linestyle='--',color='gray')
    fig.tight_layout();fig.savefig(BASE/'structural_recycle_trace.png',dpi=180);plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
