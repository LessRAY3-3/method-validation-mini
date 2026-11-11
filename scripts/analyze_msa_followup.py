"""Validate and summarize the targeted MSA outlier follow-up without inference."""
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prediction_utils import ca_models, fitted_ca_rmsd
from prediction_provenance import sha256
from analyze_predictions import validate_scores
from analyze_prediction_ablation import verify_run

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/prediction_msa_diagnosis/followup'


def inspect_run(base, plan, spec):
    folder=base/'runs'/spec['id']
    record=json.loads((folder/'manifest.json').read_text())
    if record['spec']!=spec or record['plan_sha256']!=sha256(base/'plan.json') or record['weights_sha256']!=plan['weights_sha256']:
        raise ValueError('Manifest mismatch')
    msa=base/plan['inputs'][spec['condition']]['path']
    if sha256(msa)!=record['input_sha256'] or record['input_sha256']!=plan['inputs'][spec['condition']]['sha256']:
        raise ValueError('MSA changed')
    current={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'}
    if current!=record['outputs_sha256']:raise ValueError('Output changed')
    log=(folder/'run.log').read_text()
    recycles=list(map(int,re.findall(r'recycle=(\d+)',log)))
    if recycles!=list(range(spec['recycles']+1)) or 'Skipping ' in log or 'SUBMIT:' in log:
        raise ValueError('Unexpected inference execution')
    config=json.loads((folder/'config.json').read_text())
    expected={'random_seed':spec['seed'],'num_models':1,'num_seeds':1,'num_recycles':spec['recycles'],'model_type':'alphafold2_ptm','model_order':[1],'use_templates':False,'num_relax':0,'use_dropout':False}
    if any(config[k]!=v for k,v in expected.items()):raise ValueError('Config mismatch')
    if spec['extra_args'][:1]==['--max-msa']:
        a,b=map(int,spec['extra_args'][1].split(':'))
        if (config['max_seq'],config['max_extra_seq'])!=(a,b):raise ValueError('Capacity override not applied')
    scores_paths=list(folder.glob('*scores*.json'));pdbs=list(folder.glob('*unrelaxed*.pdb'))
    if len(scores_paths)!=1 or len(pdbs)!=1:raise ValueError('Missing unique prediction')
    scores=json.loads(scores_paths[0].read_text());plddt,pae=validate_scores(scores,20)
    models=ca_models(pdbs[0],plan['sequence'])
    if len(models)!=1 or not np.allclose(models[0][2],plddt,atol=.011,rtol=0):raise ValueError('PDB/scores disagree')
    capacity=re.search(r'Setting max_seq=(\d+), max_extra_seq=(\d+)',log)
    trajectory=[{'recycle':int(i),'plddt_log_rounded':float(p),'ptm_log_rounded':float(t)} for i,p,t in re.findall(r'recycle=(\d+) pLDDT=([\d.]+) pTM=([\d.]+)',log)]
    return models[0][1],plddt,pae,scores['ptm'],{'effective_max_seq':int(capacity[1]),'effective_max_extra_seq':int(capacity[2]),'version':config['version'],'recycle_confidence':trajectory,'pdb_sha256':sha256(pdbs[0]),'coordinates':models[0][1].tolist(),'scores':scores}


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    plan=json.loads((BASE/'plan.json').read_text())
    nmr=ca_models(ROOT/'data/raw/1L2Y.pdb',plan['sequence'])
    old=ROOT/'results/prediction_ablation';oldplan=json.loads((old/'plan.json').read_text())
    if sha256(old/'plan.json') != plan['baseline_plan_sha256']:
        raise ValueError('Baseline plan changed')
    rows=[];details={};extended=[];versions=set()
    for c in oldplan['conditions_in_execution_order']:
        for seed in oldplan['seeds_in_execution_order']:
            xyz,p,_,_,meta=verify_run(old,oldplan,c,seed)
            versions.add(meta['colabfold_version'])
            values=[fitted_ca_rmsd(xyz,ref) for _,ref,_ in nmr]
            extended.append({'condition':c,'seed':seed,'median_ca_rmsd_angstrom':float(np.median(values)),'mean_plddt':float(p.mean()),'stage':'original_3_seeds'})
    for spec in plan['runs']:
        xyz,p,pae,ptm,info=inspect_run(BASE,plan,spec)
        values=[fitted_ca_rmsd(xyz,ref) for _,ref,_ in nmr]
        row={'id':spec['id'],'condition':spec['condition'],'seed':spec['seed'],'recycles':spec['recycles'],'effective_max_seq':info['effective_max_seq'],'effective_max_extra_seq':info['effective_max_extra_seq'],'mean_plddt':float(p.mean()),'mean_pae_angstrom':float(pae.mean()),'ptm_json_rounded':ptm,'model1_ca_rmsd_angstrom':values[0],'median_ca_rmsd_angstrom':float(np.median(values)),'min_ca_rmsd_angstrom':min(values),'max_ca_rmsd_angstrom':max(values)}
        rows.append(row);details[spec['id']]=info;versions.add(info['version'])
        if spec['id'].startswith('extension_'):extended.append({'condition':row['condition'],'seed':row['seed'],'median_ca_rmsd_angstrom':row['median_ca_rmsd_angstrom'],'mean_plddt':row['mean_plddt'],'stage':'prespecified_3_new_seeds'})
    if len(versions)!=1:raise ValueError('Software versions differ')
    refcoords,refp,refpae,refptm,_=verify_run(old,oldplan,'deduplicated_saved_msa',20261110)
    repeated=details['repeat_bad_seed']
    repeat={'coordinates_equal':bool(np.array_equal(refcoords,repeated['coordinates'])),'plddt_equal':bool(np.array_equal(refp,repeated['scores']['plddt'])),'pae_equal':bool(np.array_equal(refpae,repeated['scores']['pae'])),'ptm_equal':refptm==repeated['scores']['ptm']}
    rawxyz,rawp,rawpae,rawptm,_=verify_run(old,oldplan,'raw_saved_msa',20261110)
    dup=details['duplicate_homolog_bad_seed']
    duplicate_equivalence={'coordinates_equal_to_original_raw':bool(np.array_equal(rawxyz,dup['coordinates'])),'plddt_equal_to_original_raw':bool(np.array_equal(rawp,dup['scores']['plddt'])),'pae_equal_to_original_raw':bool(np.array_equal(rawpae,dup['scores']['pae']))}
    conditions=oldplan['conditions_in_execution_order']
    aggregates={}
    for c in conditions:
        rr=[r for r in extended if r['condition']==c]
        if len(rr)!=6 or len(set(r['seed'] for r in rr))!=6:raise ValueError('Six-seed panel incomplete')
        a=np.array([r['median_ca_rmsd_angstrom'] for r in rr])
        aggregates[c]={'n_seeds':6,'median_of_seed_medians_angstrom':float(np.median(a)),'min_angstrom':float(a.min()),'max_angstrom':float(a.max()),'count_above_2A_descriptive_not_universal_cutoff':int(sum(a>2))}
    summary={'all_14_followup_runs_verified':True,'plan_sha256':sha256(BASE/'plan.json'),'exact_repeat':repeat,'duplicate_homolog_equivalence':duplicate_equivalence,'targeted_runs':rows,'six_seed_panel':extended,'six_seed_summary':aggregates,'recycle_traces':{k:v['recycle_confidence'] for k,v in details.items()},'limits':plan['limits']+' This is post-hoc mechanistic follow-up, not an independent held-out target benchmark; duplicate weighting, MSA shape and stochastic processing can co-vary. No p-values or success claims from NMR model count.'}
    write_csv(BASE/'targeted_results.csv',rows);write_csv(BASE/'six_seed_panel.csv',extended)
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for seed in sorted(set(r['seed'] for r in extended)):
        rr=[next(r for r in extended if r['condition']==c and r['seed']==seed) for c in conditions]
        axes[0].plot(range(3),[r['median_ca_rmsd_angstrom'] for r in rr],marker='o',label=str(seed))
    axes[0].set(xticks=range(3),xticklabels=['Query only','Deduplicated','Raw saved'],ylabel='Median CA RMSD to NMR ensemble (Å)',title='Six paired seeds, same model and CPU route')
    axes[0].legend(fontsize=7)
    target=rows[:5]
    axes[1].bar(range(5),[r['median_ca_rmsd_angstrom'] for r in target],color=['#a44','#288','#288','#c93','#446'])
    axes[1].set(xticks=range(5),xticklabels=['Exact\nrepeat','Duplicate\nquery','Duplicate\nhomolog','Raw cap\n2:1','Dedup\n12 recycles'],ylabel='Median CA RMSD (Å)',title='Targeted perturbations: seed 20261110')
    fig.tight_layout();fig.savefig(BASE/'followup_overview.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    trace=details['dedup_12_recycles_bad_seed']['recycle_confidence']
    ax.plot([r['recycle'] for r in trace],[r['plddt_log_rounded'] for r in trace],marker='o')
    ax.axvline(3,color='gray',linestyle='--',label='Original recycle limit')
    ax.set(xlabel='Recycle index',ylabel='Mean pLDDT (rounded in log)',title='Longer refinement of the same outlier; confidence is not accuracy');ax.legend();fig.tight_layout();fig.savefig(BASE/'recycle_trace.png',dpi=180);plt.close(fig)
    print(json.dumps({'repeat':repeat,'duplicate_homolog':duplicate_equivalence,'summary':aggregates,'targeted':rows[:5]},indent=2))


if __name__=='__main__':main()
