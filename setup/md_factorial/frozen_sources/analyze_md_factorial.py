#!/usr/bin/env python3
"""Descriptive 2x2 MD analysis; whole-protein extraction uses each run's own TPR.

No simulation. All cells and seeds are required. Branches share history, so
factorial contrasts are descriptive, without frame-level significance tests.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from analyze_md_replicates import (sha, read_gro_trajectory, validate_times,
    window_selection, fitted_coordinates, geometry_metrics, load_xvg)
from analyze_md_confirmation_batch import episodes, selected_occupancy, write_csv
from prediction_utils import ca_models, fasta_sequence, AA

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261113, 20261114, 20261115)
WINDOWS = ((0, 11), (1, 3), (8, 11))
CELLS = ('A', 'B', 'C', 'D')


def gro_identities(path, count=304):
    lines = Path(path).read_text().splitlines()
    if int(lines[1]) < count:
        raise ValueError('Missing protein atoms in reference')
    return [(int(s[:5]), s[5:10].strip(), s[10:15].strip()) for s in lines[2:2+count]]


def contrasts(rows):
    """Three per-seed branch contrasts, never pretend matched labels ensure independence."""
    index = {(r['cell'], r['seed'], r['window_ns'], r['metric']): r['value'] for r in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate endpoint rows')
    keys = sorted({(r['window_ns'], r['metric']) for r in rows})
    result = []
    for window, metric in keys:
        for seed in SEEDS:
            try:
                a,b,c,d = (index[(cell, seed, window, metric)] for cell in CELLS)
            except KeyError as exc:
                raise ValueError('Missing factorial cell/seed endpoint') from exc
            for name,value in [('B-A',b-a),('D-C',d-c),('C-A',c-a),('D-B',d-b),('(D-C)-(B-A)',(d-c)-(b-a))]:
                result.append(dict(seed=seed, window_ns=window, metric=metric, contrast=name, value=value))
    return result


def validate_completed(directory, seed):
    for name in ('md.done.json', 'provenance.json', 'md.tpr', 'md.xtc', 'md.log'):
        if not (directory/name).is_file():
            raise ValueError(f'Missing {directory/name}')
    done=json.loads((directory/'md.done.json').read_text())
    provenance=json.loads((directory/'provenance.json').read_text())
    if done.get('nsteps') != 5500000 or provenance.get('seed') != seed or provenance.get('target_production_ns') != 11:
        raise ValueError(f'Incomplete or wrong run: {directory}')
    log=(directory/'md.log').read_text()
    if 'LINCS WARNING' in log or 'Fatal error:' in log:
        raise ValueError(f'Numerical/technical failure: {directory}')
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'results/md_replicates_gpu_confirmation')
    parser.add_argument('--factorial-root',type=Path,default=ROOT/'results/md_factorial')
    parser.add_argument('--output', '--output-dir', dest='output', type=Path)
    parser.add_argument('--gmx',default='gmx')
    args=parser.parse_args()
    output=(args.output or args.factorial_root/'analysis').resolve()
    if output.exists(): raise ValueError('Output exists; never overwrite evidence')
    refs=ca_models(ROOT/'data/raw/1L2Y.pdb',fasta_sequence(ROOT/'data/prepared/1L2Y.fasta'))
    if len(refs)!=38: raise ValueError('Require all 38 NMR references')
    sources=[ROOT/'results/md/md.tpr',ROOT/'results/md/minim.gro',ROOT/'data/raw/1L2Y.pdb',ROOT/'data/prepared/1L2Y.fasta']
    runs=[]
    for cell in CELLS:
        for seed in SEEDS:
            directory=((args.baseline if cell=='A' else args.factorial_root/cell)/f'seed_{seed}').resolve()
            provenance=validate_completed(directory,seed)
            runs.append((cell,seed,directory,provenance))
            sources.extend(directory/name for name in ('md.tpr','md.xtc','md.log','md.done.json','provenance.json'))
    inputs={str(p):sha(p) for p in sources}
    output.mkdir(parents=True)
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':inputs,
        'sources':{name:sha(ROOT/'scripts'/name) for name in ('analyze_md_factorial.py','analyze_md_replicates.py','analyze_md_confirmation_batch.py','prediction_utils.py')},
        'windows_ns':WINDOWS,'seeds':SEEDS,'cells':CELLS,'analysis_gromacs':subprocess.check_output([args.gmx,'--version'],text=True),
        'classification':'Prespecified conditional sensitivity experiment; analysis-time manifest is not launch-time protocol',
        'limitations':['n=3 seed branches per cell, correlated frames are not replicates', 'Short/long NPT branches share history; identical seed labels across starting systems do not imply identical random streams', 'Two NMR starting structures do not establish arbitrary-start robustness or equilibrium', 'No p-values, folding rates, lifetimes, convergence or thermodynamic probabilities']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    refids=gro_identities(ROOT/'results/md/minim.gro')
    endpoint_rows=[]; threshold_rows=[]; episode_rows=[]; reference_rows=[]; q_rows=[]; contact_rows=[]; rmsf_rows=[]
    series={}; commands=[]; check_errors=[]
    i,j=np.triu_indices(20,k=3)
    native=np.linalg.norm(refs[0][1][i]-refs[0][1][j],axis=-1)
    for cell,seed,directory,provenance in runs:
        dest=output/cell/f'seed_{seed}';dest.mkdir(parents=True)
        def run(argv,selection):
            command=[args.gmx]+argv
            p=subprocess.run(command,input=selection,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,cwd=dest)
            commands.append({'cell':cell,'seed':seed,'argv':command,'selection':selection,'returncode':p.returncode})
            with (dest/'analysis.log').open('a') as h:h.write(p.stdout+'\n')
            if p.returncode:raise RuntimeError(f'Analysis failed: {dest}')
        whole=dest/'protein_whole.gro'
        run(['trjconv','-s',str(directory/'md.tpr'),'-f',str(directory/'md.xtc'),'-o',str(whole),'-pbc','whole','-b','0','-e','11000'],'Protein\n')
        times,xyz,ids=read_gro_trajectory(whole);validate_times(times,11)
        if ids!=refids:raise ValueError('Protein atom identities/order differ from common reference; cannot compare safely')
        metrics,aligned,_=geometry_metrics(xyz,ids,refs[0][1])
        run(['rms','-s',str(ROOT/'results/md/md.tpr'),'-f',str(whole),'-o','rmsd.xvg','-tu','ns'],'Backbone\nBackbone\n')
        run(['gyrate','-s',str(directory/'md.tpr'),'-f',str(whole),'-o','gyrate.xvg'],'Protein\n')
        run(['hbond','-s',str(directory/'md.tpr'),'-f',str(whole),'-num','hbond.xvg','-o','hbond.ndx'],'Protein\nProtein\n')
        metrics.update(backbone_rmsd_common_reference_nm=load_xvg(dest/'rmsd.xvg',11,ns=True),gyrate_nm=load_xvg(dest/'gyrate.xvg',11),protein_internal_hbonds=load_xvg(dest/'hbond.xvg',11))
        ca=xyz[:,[k for k,(_,_,name) in enumerate(ids) if name=='CA']]
        distances=np.linalg.norm(ca[:,i]-ca[:,j],axis=-1)
        core=fitted_coordinates(ca[:,2:18],refs[0][1][2:18])
        metrics['core_ca_rmsd_nmr1_angstrom']=np.sqrt(np.mean(np.sum((core-refs[0][1][2:18])**2,axis=-1),axis=-1))
        core_pairs=(i>=2)&(j<18)
        metrics['core_nonlocal_distance_mae_nmr1_angstrom']=abs(distances[:,core_pairs]-native[core_pairs]).mean(axis=1)
        metrics['y3p19_fraction_above_6A']=(metrics['sidechain_min_3_19_angstrom']>6).astype(float)
        series[(cell,seed)]=metrics
        write_csv(dest/'timeseries.csv',[{'time_ps':float(t),**{k:float(v[n]) for k,v in metrics.items()}} for n,t in enumerate(times)])
        for a,b in WINDOWS:
            mask=window_selection(times,a,b,11); window=f'{a}-{b}';base={'cell':cell,'seed':seed,'window_ns':window}
            endpoint_rows.extend({**base,'metric':k,'value':float(v[mask].mean())} for k,v in metrics.items())
            rmsf=np.sqrt(np.mean(np.sum((aligned[mask]-aligned[mask].mean(axis=0))**2,axis=-1),axis=0))/10
            rmsf_rows.extend({**base,'residue':k+1,'ca_rmsf_nm':float(v)} for k,v in enumerate(rmsf))
            for threshold in (5,6,7):
                above=metrics['sidechain_min_3_19_angstrom'][mask]>threshold
                spans=episodes(times[mask],above)
                threshold_rows.append({**base,'threshold_A':threshold,'sample_fraction_above':float(above.mean()),'n_samples_above':int(above.sum()),'longest_sampled_span_ps':max((end-start for start,end,n in spans),default=0)})
                episode_rows.extend({**base,'threshold_A':threshold,'first_sample_ps':start,'last_sample_ps':end,'n_samples':n,'sampled_span_ps':end-start} for start,end,n in spans)
            for cutoff in (7,8,9):
                selected=native<=cutoff
                for factor in (1.1,1.2,1.3):
                    q=np.mean(distances[mask][:,selected]<=factor*native[selected],axis=1)
                    q_rows.append({**base,'native_cutoff_A':cutoff,'retention_factor':factor,'native_pair_count':int(selected.sum()),'mean_q':float(q.mean())})
                    for first,second in ((3,19),(12,18)):
                        index=int(np.flatnonzero((i==first-1)&(j==second-1))[0])
                        status,value=selected_occupancy(native[index],distances[mask,index],cutoff,factor)
                        contact_rows.append({**base,'native_cutoff_A':cutoff,'retention_factor':factor,'residue_i':first,'residue_j':second,'selection_status':status,'occupancy':value})
        for number,ref,_ in refs:
            aligned_ref=fitted_coordinates(ca,ref)
            rmsd=np.sqrt(np.mean(np.sum((aligned_ref-ref)**2,axis=-1),axis=-1))
            irmsd=np.sqrt(np.mean(np.sum((fitted_coordinates(ca[:,2:18],ref[2:18])-ref[2:18])**2,axis=-1),axis=-1))
            refdist=np.linalg.norm(ref[i]-ref[j],axis=-1)
            mae=abs(distances-refdist).mean(axis=1)
            coremae=abs(distances[:,core_pairs]-refdist[core_pairs]).mean(axis=1)
            for k in (0,150,1100):
                svd=SVDSuperimposer();svd.set(ref,ca[k]);svd.run()
                error=abs(float(rmsd[k])-svd.get_rms());check_errors.append(error)
                if error>1e-10:raise ValueError('SVD verification failed')
            for a,b in WINDOWS:
                mask=window_selection(times,a,b,11)
                reference_rows.append({'cell':cell,'seed':seed,'window_ns':f'{a}-{b}','nmr_model':number,'mean_ca_rmsd_angstrom':float(rmsd[mask].mean()),'mean_interior_ca_rmsd_angstrom':float(irmsd[mask].mean()),'mean_nonlocal_distance_mae_angstrom':float(mae[mask].mean()),'mean_core_nonlocal_distance_mae_angstrom':float(coremae[mask].mean())})
    contrast_rows=contrasts(endpoint_rows)
    for name,rows in [('per_trajectory_windows',endpoint_rows),('threshold_sensitivity',threshold_rows),('all_reference_window_results',reference_rows),('all_q_windows',q_rows),('selected_contact_occupancies',contact_rows),('per_residue_rmsf',rmsf_rows),('seed_contrasts',contrast_rows)]:
        write_csv(output/f'{name}.csv',rows)
    write_csv(output/'threshold_episodes.csv',episode_rows,['cell','seed','window_ns','threshold_A','first_sample_ps','last_sample_ps','n_samples','sampled_span_ps'])
    summary_rows=[]
    for label,rows,groupkey in [('cell',endpoint_rows,'cell'),('contrast',contrast_rows,'contrast')]:
        for group,window,metric in sorted({(r[groupkey],r['window_ns'],r['metric']) for r in rows}):
            values=[r['value'] for r in rows if (r[groupkey],r['window_ns'],r['metric'])==(group,window,metric)]
            if len(values)!=3:raise ValueError('Expected three branch values')
            summary_rows.append({'kind':label,'group':group,'window_ns':window,'metric':metric,'n_seed_branches':3,'mean':float(np.mean(values)),'minimum':min(values),'maximum':max(values)})
    write_csv(output/'descriptive_summary.csv',summary_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,4,figsize=(16,9),sharex=True,sharey='row')
    for col,cell in enumerate(CELLS):
        for row,(key,label) in enumerate([('backbone_rmsd_common_reference_nm','Backbone RMSD (nm)'),('sidechain_min_3_19_angstrom','Y3-P19 sidechain min (A)'),('core_ca_rmsd_nmr1_angstrom','Core CA RMSD / NMR1 (A)')]):
            ax=axes[row,col]
            for seed in SEEDS:ax.plot(times/1000,series[(cell,seed)][key],label=str(seed),lw=.65)
            ax.axvspan(1,3,color='orange',alpha=.1);ax.axvspan(8,11,color='grey',alpha=.1)
            if col==0:ax.set_ylabel(label)
            if row==0:ax.set_title({'A':'A: model1 / 100ps NPT','B':'B: model1 / 1000ps NPT','C':'C: model24 / 100ps NPT','D':'D: model24 / 1000ps NPT'}[cell])
            if row==2:ax.set_xlabel('Time after restraint release (ns)')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(output/'factorial_trajectories.png',dpi=180);plt.close(fig)
    if any(sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('Inputs changed during analysis')
    (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    summary={'status':'complete','n_cells':4,'n_seed_branches_per_cell':3,'frames_per_run':1101,'independent_svd_checks':len(check_errors),'maximum_svd_error_A':max(check_errors),'limitations':manifest['limitations'],'outputs_sha256':{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'status':'complete','output':str(output)}))

if __name__=='__main__':main()
