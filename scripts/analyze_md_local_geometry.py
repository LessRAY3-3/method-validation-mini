"""Post-hoc local geometry diagnosis of all three fixed MD trajectories; no dynamics."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from analyze_md_replicates import read_gro_trajectory, validate_times, window_selection, fitted_coordinates
from prediction_utils import ca_models, fasta_sequence
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(20261110,20261111,20261112)
WINDOWS=((0,1),(1,3),(3,8),(8,11),(0,11))
OUT=ROOT/'results/md_deep_dive/local_geometry'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_geometry(ca, reference):
    """Nonlocal CA-pair distance error, independent of rigid fitting (Å)."""
    i,j=np.triu_indices(len(reference),k=3)
    native=np.linalg.norm(reference[i]-reference[j],axis=-1)
    distances=np.linalg.norm(ca[:,i]-ca[:,j],axis=-1)
    signed=distances-native
    return i,j,native,distances,signed


def rmsd_series(ca, reference, indices):
    target=reference[indices]
    fit=fitted_coordinates(ca[:,indices],target)
    return np.sqrt(np.mean(np.sum((fit-target)**2,axis=-1),axis=-1))


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    if OUT.exists():
        raise ValueError('Refusing to overwrite an existing diagnosis')
    OUT.mkdir(parents=True)
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/raw/1L2Y.pdb',*[ROOT/f'results/md_replicates_gpu/aggregate_analysis_audited/seed_{s}/protein_whole.gro' for s in SEEDS]]}
    plan={'declared_utc':datetime.now(timezone.utc).isoformat(),'classification':'Exploratory post-hoc diagnosis after observing seed20261110 1–3ns deviation; not independent confirmatory evidence','seeds':SEEDS,'windows_ns':WINDOWS,'window_rule':'Left inclusive, right exclusive except 11ns included','reference':'NMR model1; all-CA and independently fitted CA3–18; nonlocal pair-distance errors are fitting-independent','pair_selection':'All CA pairs with residue separation>=3; internal subset both residues3–18','contact_definition':'Reference cutoff7/8/9A, trajectory<=1.2*reference distance','reporting':'All pair/seed/window values retained; top ranks descriptive only, no frame-level hypothesis tests','input_sha256':inputs,'script_sha256':sha(__file__)}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    sequence=fasta_sequence(ROOT/'data/prepared/1L2Y.fasta')
    ref=ca_models(ROOT/'data/raw/1L2Y.pdb',sequence)[0][1]
    windows=[];pair_rows=[];residue_rows=[];contacts=[];allseries={};heat={};checks=[]
    for seed in SEEDS:
        p=ROOT/f'results/md_replicates_gpu/aggregate_analysis_audited/seed_{seed}/protein_whole.gro'
        times,xyz,ids=read_gro_trajectory(p);validate_times(times,11)
        ix=[k for k,x in enumerate(ids) if x[2]=='CA']
        assert [ids[k][0] for k in ix]==list(range(1,21))
        ca=xyz[:,ix];i,j,native,distances,signed=pair_geometry(ca,ref)
        error=np.abs(signed);internal=(i>=2)&(j<=17)
        series={'all_ca_rmsd_A':rmsd_series(ca,ref,np.arange(20)),'internal3_18_refit_rmsd_A':rmsd_series(ca,ref,np.arange(2,18)),'nonlocal_pair_mae_A':error.mean(axis=1),'internal3_18_pair_mae_A':error[:,internal].mean(axis=1)}
        for cutoff in (7,8,9):series[f'q{cutoff}']=np.mean(distances[:,native<=cutoff]<=1.2*native[native<=cutoff],axis=1)
        allseries[seed]=series
        for frame in (0,150,1100):
            for subset,label in [(np.arange(20),'all'),(np.arange(2,18),'internal')]:
                fit=SVDSuperimposer();fit.set(ref[subset],ca[frame,subset]);fit.run()
                err=abs(fit.get_rms()-series['all_ca_rmsd_A' if label=='all' else 'internal3_18_refit_rmsd_A'][frame])
                assert err<1e-10
                checks.append({'seed':seed,'frame':frame,'subset':label,'independent_rmsd_error_A':float(err)})
            for a,b in [(2,18),(5,11),(7,14)]:
                match=np.flatnonzero((i==a)&(j==b))[0]
                d=sum((float(ca[frame,a,k])-float(ca[frame,b,k]))**2 for k in range(3))**.5
                assert abs(d-distances[frame,match])<1e-10
        for start,end in WINDOWS:
            mask=window_selection(times,start,end,11);label=f'{start}-{end}'
            windows.append({'seed':seed,'window_ns':label,'frames':int(mask.sum()),**{k:float(v[mask].mean()) for k,v in series.items()}})
            pair_mae=error[mask].mean(axis=0);pair_signed=signed[mask].mean(axis=0)
            for k,(a,b) in enumerate(zip(i,j)):
                pair_rows.append({'seed':seed,'window_ns':label,'residue_i':int(a+1),'residue_j':int(b+1),'reference_distance_A':float(native[k]),'mean_distance_A':float(distances[mask,k].mean()),'mean_signed_error_A':float(pair_signed[k]),'mean_absolute_error_A':float(pair_mae[k])})
            for residue in range(20):
                selected=(i==residue)|(j==residue)
                residue_rows.append({'seed':seed,'window_ns':label,'residue':residue+1,'aa':sequence[residue],'nonlocal_partner_count':int(selected.sum()),'mean_nonlocal_distance_error_A':float(pair_mae[selected].mean())})
            for cutoff in (7,8,9):
                for k in np.flatnonzero(native<=cutoff):
                    contacts.append({'seed':seed,'window_ns':label,'native_cutoff_A':cutoff,'residue_i':int(i[k]+1),'residue_j':int(j[k]+1),'reference_distance_A':float(native[k]),'retention_threshold_A':float(1.2*native[k]),'occupancy':float(np.mean(distances[mask,k]<=1.2*native[k]))})
            matrix=np.full((20,20),np.nan);matrix[i,j]=pair_mae;matrix[j,i]=pair_mae;heat[seed,label]=matrix
    write_csv('window_summary.csv',windows);write_csv('all_pair_distance_errors.csv',pair_rows);write_csv('per_residue_distance_errors.csv',residue_rows);write_csv('all_contact_occupancies.csv',contacts)
    contrasts=[]
    for seed in SEEDS:
        for cutoff in (7,8,9):
            early=[r for r in contacts if r['seed']==seed and r['native_cutoff_A']==cutoff and r['window_ns']=='1-3']
            late={(r['residue_i'],r['residue_j']):r for r in contacts if r['seed']==seed and r['native_cutoff_A']==cutoff and r['window_ns']=='8-11'}
            for r in early:
                occupancy=late[r['residue_i'],r['residue_j']]['occupancy']
                contrasts.append({'seed':seed,'native_cutoff_A':cutoff,'residue_i':r['residue_i'],'residue_j':r['residue_j'],'early_occupancy':r['occupancy'],'late_occupancy':occupancy,'late_minus_early':occupancy-r['occupancy']})
    write_csv('all_contact_early_late_contrasts.csv',contrasts)
    top_residues={str(s):sorted([r for r in residue_rows if r['seed']==s and r['window_ns']=='1-3'],key=lambda r:r['mean_nonlocal_distance_error_A'],reverse=True)[:6] for s in SEEDS}
    top_pairs={str(s):sorted([r for r in pair_rows if r['seed']==s and r['window_ns']=='1-3'],key=lambda r:r['mean_absolute_error_A'],reverse=True)[:8] for s in SEEDS}
    top_contacts={str(s):sorted([r for r in contrasts if r['seed']==s and r['native_cutoff_A']==8],key=lambda r:r['late_minus_early'],reverse=True)[:6] for s in SEEDS}
    for p,h in inputs.items():assert sha(ROOT/p)==h,'Input changed during analysis'
    result={'status':'completed_exploratory_diagnosis','input_sha256':inputs,'plan_sha256':sha(OUT/'plan.json'),'windows':windows,'top_early_residue_error_descriptive_only':top_residues,'top_early_pair_error_descriptive_only':top_pairs,'top_contact_recovery_descriptive_only':top_contacts,'independent_checks':checks,'limitations':['Windows chosen after observing trajectories, not a confirmatory independent test','All three runs start from the same minimized structure; 11ns cannot establish thermodynamic convergence','Contact changes and RMSD co-occurrence are geometric association, not causal mechanism','Individual frames and NMR models are not independent trajectory replicates','Mean pair errors weight residue partners equally; terminal residues have different partner counts','GRO coordinate precision0.001nm; extra decimals facilitate verification, not physical precision']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    for col,seed in enumerate(SEEDS):
        for row,label in enumerate(('1-3','8-11')):
            ax=axes[row,col];im=ax.imshow(heat[seed,label],origin='lower',vmin=0,vmax=5,cmap='magma',extent=(.5,20.5,.5,20.5));ax.set_title(f'{seed}: {label} ns');ax.set_xticks([1,5,10,15,20]);ax.set_yticks([1,5,10,15,20]);ax.set_xlabel('Residue');ax.set_ylabel('Residue')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Mean absolute CA pair-distance error (Å)',shrink=.8);fig.savefig(OUT/'pair_error_heatmaps.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for seed,series in allseries.items():
        axes[0].plot(times/1000,series['internal3_18_refit_rmsd_A'],label=str(seed),lw=.8)
        axes[1].plot(times/1000,series['internal3_18_pair_mae_A'],label=str(seed),lw=.8)
    for ax in axes:ax.axvspan(1,3,color='gray',alpha=.1);ax.legend(fontsize=8)
    axes[0].set_ylabel('CA3–18 independently fitted RMSD (Å)');axes[1].set_ylabel('CA3–18 nonlocal distance MAE (Å)');axes[1].set_xlabel('Time (ns)');fig.tight_layout();fig.savefig(OUT/'internal_geometry_timeseries.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
    for ax,seed in zip(axes,SEEDS):
        for label in ('1-3','8-11'):
            rows=[r for r in residue_rows if r['seed']==seed and r['window_ns']==label]
            ax.plot([r['residue'] for r in rows],[r['mean_nonlocal_distance_error_A'] for r in rows],marker='.',label=label+' ns')
        ax.set_title(str(seed));ax.set_xlabel('Residue');ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean nonlocal CA distance error (Å)');fig.tight_layout();fig.savefig(OUT/'residue_error_profiles.png',dpi=180);plt.close(fig)
    print(json.dumps({'status':result['status'],'output':str(OUT)},indent=2))


if __name__=='__main__':main()
