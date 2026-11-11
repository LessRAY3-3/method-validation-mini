"""Compare completed original/new MD batches descriptively, never starts dynamics.

Requires all real new-seed results and verifies every manifest-listed output.
Original and new batches remain separate; a six-run pooled view is secondary.
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=(20261110,20261111,20261112)
NEW=(20261113,20261114,20261115)
WINDOWS=('0-11','1-3','8-11')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_files(root, hashes):
    if not hashes:raise ValueError('Missing file hashes')
    for name,want in hashes.items():
        p=root/name
        if not p.resolve().is_relative_to(root.resolve()) or not p.is_file():
            raise ValueError(f'Missing or invalid evidence path: {name}')
        if sha(p)!=want:raise ValueError(f'Evidence hash mismatch: {name}')


def load_verified_batch(path,new):
    summary=json.loads((path/'summary.json').read_text())
    expected=NEW if new else OLD
    wanted='three_completed_confirmation_replicates_verified' if new else 'original_seed_method_verification_passed'
    if summary.get('status')!=wanted or tuple(summary.get('seeds',()))!=expected or summary.get('n_trajectories')!=3 or summary.get('frames_per_trajectory')!=1101:
        raise ValueError('Batch incomplete, incorrect seed set or wrong study status')
    if new:
        verify_files(path,summary.get('output_sha256',{}))
    else:
        copy=json.loads((path/'copy_verification.json').read_text())
        if copy.get('status')!='top_level_files_copied_and_hash_verified':raise ValueError('Original copy unverified')
        verify_files(path,copy['sha256'])
    return summary


def read_checked(path,seeds,per_seed_count=None):
    with path.open() as f:rows=list(csv.DictReader(f))
    if {int(r['seed']) for r in rows}!=set(seeds):raise ValueError(f'Wrong CSV seeds: {path}')
    if {r['window_ns'] for r in rows}!=set(WINDOWS):raise ValueError(f'Wrong CSV windows: {path}')
    if per_seed_count is not None:
        for seed in seeds:
            if sum(int(r['seed'])==seed for r in rows)!=per_seed_count:raise ValueError('Missing or duplicated CSV records')
    return rows



def validate_contacts(rows,seeds):
    from itertools import product
    expected=set(product(seeds,WINDOWS,(7.,8.,9.),(1.1,1.2,1.3),((3,19),(12,18))))
    seen=set()
    for r in rows:
        key=(int(r['seed']),r['window_ns'],float(r['native_cutoff_A']),float(r['retention_factor']),(int(r['residue_i']),int(r['residue_j'])))
        if key in seen:raise ValueError('Duplicate selected contact record')
        seen.add(key)
        reference=float(r['reference_distance_A'])
        if not np.isfinite(reference):raise ValueError('Nonfinite reference distance')
        excluded=reference>key[2]
        if excluded:
            if r['selection_status']!='excluded' or r['occupancy']!='':raise ValueError('Excluded contact must have null occupancy')
        elif r['selection_status']!='included' or r['occupancy']=='' or not 0<=float(r['occupancy'])<=1:
            raise ValueError('Invalid included contact occupancy')
    if seen!=expected:raise ValueError('Missing or unexpected selected contact records')


def summarize(values):
    x=np.asarray(values,dtype=float)
    if not np.isfinite(x).all() or len(x)<2:raise ValueError('Nonfinite or insufficient values')
    return dict(n_trajectories=len(x),mean=float(x.mean()),between_trajectory_sd=float(x.std(ddof=1)),minimum=float(x.min()),maximum=float(x.max()))


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-root',type=Path,default=ROOT/'results/md_replicates_gpu_confirmation')
    p.add_argument('--old-analysis',type=Path,default=ROOT/'results/md_confirmation_preparation/original_batch_same_analysis')
    p.add_argument('--old-series-root',type=Path,default=ROOT/'results/md_replicates_gpu/aggregate_analysis_audited')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/md_confirmation_comparison')
    args=p.parse_args();out=args.output_dir.resolve()
    if out.exists():raise ValueError('Output exists; refusing overwrite')
    new=args.new_root.resolve()/'confirmation_analysis';old=args.old_analysis.resolve()
    new_summary=load_verified_batch(new,True);old_summary=load_verified_batch(old,False)
    sources=[new/'summary.json',old/'summary.json',old/'copy_verification.json']
    records=[]; contact_records=[]
    def add(batch,seed,window,family,metric,value,definition=''):
        v=float(value)
        if not np.isfinite(v):raise ValueError('Nonfinite comparison value')
        records.append(dict(batch=batch,seed=int(seed),window_ns=window,family=family,metric=metric,definition=definition,value=v))
    for batch,path,seeds in [('original',old,OLD),('confirmation',new,NEW)]:
        for filename in ['per_trajectory_windows.csv','threshold_sensitivity.csv','all_reference_window_results.csv','all_q_windows.csv','selected_contact_occupancies.csv']:
            sources.append(path/filename)
        rows=read_checked(path/'per_trajectory_windows.csv',seeds,3)
        if len({(r['seed'],r['window_ns']) for r in rows})!=9:raise ValueError('Duplicate seed/window')
        for r in rows:
            expected=200 if r['window_ns']=='1-3' else 1101 if r['window_ns']=='0-11' else 301
            if int(r['frames'])!=expected:raise ValueError('Wrong window frame count')
            for key,val in r.items():
                if key not in ('seed','window_ns','frames'):add(batch,r['seed'],r['window_ns'],'observable',key,val)
        rows=read_checked(path/'threshold_sensitivity.csv',seeds,9)
        if len({(r['seed'],r['window_ns'],r['threshold_A']) for r in rows})!=27:raise ValueError('Duplicate threshold row')
        for r in rows:
            if float(r['threshold_A']) not in (5,6,7):raise ValueError('Unknown distance threshold')
            for metric in ('sample_fraction_above','longest_sampled_span_ps','n_sampled_episodes'):
                add(batch,r['seed'],r['window_ns'],'Y3_P19_threshold',metric,r[metric],'distance>'+r['threshold_A']+'A')
        rows=read_checked(path/'all_reference_window_results.csv',seeds,114)
        groups=defaultdict(list)
        for r in rows:groups[r['seed'],r['window_ns']].append(r)
        for (seed,window),group in groups.items():
            if sorted(int(r['nmr_model']) for r in group)!=list(range(1,39)):raise ValueError('Missing/duplicate NMR references')
            for metric in ('mean_ca_rmsd_angstrom','mean_interior_ca_rmsd_angstrom','mean_nonlocal_distance_mae_angstrom'):
                x=np.array([float(r[metric]) for r in group])
                for stat,val in [('min',x.min()),('median',np.median(x)),('max',x.max())]:
                    add(batch,seed,window,'NMR_reference_choice',metric+'_'+stat,val,'38 references; range is not CI')
        rows=read_checked(path/'all_q_windows.csv',seeds,27)
        keys={(r['seed'],r['window_ns'],r['native_cutoff_A'],r['retention_factor']) for r in rows}
        if len(keys)!=81:raise ValueError('Duplicate Q definition')
        for r in rows:
            if float(r['native_cutoff_A']) not in (7,8,9) or float(r['retention_factor']) not in (1.1,1.2,1.3):raise ValueError('Unknown Q definition')
            add(batch,r['seed'],r['window_ns'],'Q_definition','mean_q',r['mean_q'],f"native<={r['native_cutoff_A']}A;retention<={r['retention_factor']}*reference")
        rows=read_checked(path/'selected_contact_occupancies.csv',seeds,54)
        validate_contacts(rows,seeds)
        for r in rows:
            contact_records.append({'batch':batch,**r})
            definition=f"pair={r['residue_i']}-{r['residue_j']};native<={r['native_cutoff_A']}A;retention<={r['retention_factor']}*reference"
            if r['selection_status']=='included':
                add(batch,r['seed'],r['window_ns'],'selected_CA_contact','occupancy',r['occupancy'],definition)
    # All per-run measurements retained; descriptive pooled rows do not erase batch.
    statrows=[]
    for view,wanted in [('original',OLD),('confirmation',NEW),('combined_descriptive_two_batches',OLD+NEW)]:
        groups=defaultdict(list)
        for r in records:
            if r['seed'] in wanted:groups[r['window_ns'],r['family'],r['metric'],r['definition']].append(r)
        for (window,family,metric,definition),group in groups.items():
            if sorted(r['seed'] for r in group)!=list(wanted):raise ValueError('Incomplete comparison cell')
            statrows.append(dict(view=view,window_ns=window,family=family,metric=metric,definition=definition,**summarize([r['value'] for r in group]),seed_values_json=json.dumps({str(r['seed']):r['value'] for r in group},sort_keys=True)))
    series={}
    for batch,root,seeds,summary in [('original',args.old_series_root,OLD,old_summary),('confirmation',new/'aggregate_analysis',NEW,new_summary)]:
        for seed in seeds:
            path=root/f'seed_{seed}/timeseries.csv';sources.append(path)
            key=f'aggregate_analysis/seed_{seed}/timeseries.csv'
            if sha(path)!=summary['output_sha256'].get(key):raise ValueError('Time-series evidence mismatch')
            with path.open() as f:rows=list(csv.DictReader(f))
            times=np.array([float(r['time_ps']) for r in rows])
            if not np.array_equal(times,np.arange(0,11001,10)):raise ValueError('Incomplete time series')
            series[batch,seed]=(times,rows)
    hashes={str(path.resolve()):sha(path) for path in sources}
    out.mkdir(parents=True)
    write(out/'all_per_trajectory_values.csv',records)
    write(out/'all_selected_contact_records.csv',contact_records)
    write(out/'batch_and_combined_descriptive_summary.csv',statrows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
    metrics=('backbone_rmsd_common_reference_nm','q_ca_8A','sidechain_min_3_19_angstrom')
    labels=('Shared-reference backbone RMSD (nm)','Geometric Q (8A / 1.2 factor)','Y3-P19 sidechain minimum distance (A)')
    for (batch,seed),(times,rows) in series.items():
        for ax,metric,label in zip(axs,metrics,labels):
            vals=np.array([float(r[metric]) for r in rows])
            # Display complete 0.1ns blocks, exclude final singleton; CSV retains it.
            means=vals[:1100].reshape(110,10).mean(axis=1)
            ax.plot(np.arange(110)*.1+.05,means,ls='--' if batch=='original' else '-',lw=1.2,label=f'{batch}: {seed}')
            ax.set_ylabel(label)
    axs[0].legend(ncol=2,fontsize=8)
    for ax in axs:ax.axvspan(1,3,color='gray',alpha=.08);ax.axvspan(8,11,color='teal',alpha=.04)
    axs[-1].set_xlabel('Production time (ns); 0.1ns complete blocks shown')
    fig.suptitle('Original vs newly seeded batch: dashed vs solid; no pooling of frames')
    fig.tight_layout();fig.savefig(out/'six_trajectory_comparison.png',dpi=180);plt.close(fig)
    if any(sha(Path(path))!=value for path,value in hashes.items()):raise ValueError('Evidence changed during comparison')
    metadata={'status':'complete_descriptive_comparison','completed_utc':datetime.now(timezone.utc).isoformat(),'original_seeds':OLD,'confirmation_seeds':NEW,'n_trajectories_per_batch':3,'n_combined_trajectories':6,'input_sha256':hashes,'script_sha256':sha(Path(__file__)),'limitations':['New batch separately reported before six-run descriptive combination.','Both batches share one minimized starting configuration.','No p-values or binary replication-success label.','Selected contacts excluded by reference cutoff are retained with empty occupancy in all_selected_contact_records.csv; only included contacts enter statistics, never imputed zero.','NMR range measures reference-choice sensitivity, not confidence intervals.','Episodes are consecutive saved-sample spans, not exact state lifetimes.','Frames and windows are correlated; n is trajectories.'],'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'status':metadata['status'],'output':str(out)}))

if __name__=='__main__':main()
