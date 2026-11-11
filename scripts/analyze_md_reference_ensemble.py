"""Explore reference-choice sensitivity for the three existing MD trajectories.
No dynamics. All 38 NMR models are retained; they are not independent samples.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from analyze_md_replicates import read_gro_trajectory, fitted_coordinates, validate_times, sha
from prediction_utils import ca_models, fasta_sequence

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261110, 20261111, 20261112)
WINDOWS = ((0,1),(1,3),(3,8),(8,11),(0,11))

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    out=ROOT/'results/md_deep_dive/reference_ensemble'
    if out.exists():raise ValueError('Use a new result directory; existing evidence is retained')
    source=ROOT/'results/md_replicates_gpu/aggregate_analysis_audited'
    refpath=ROOT/'data/raw/1L2Y.pdb'
    inputs=[refpath]+[source/f'seed_{s}/protein_whole.gro' for s in SEEDS]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    out.mkdir(parents=True)
    protocol={'declared_utc':datetime.now(timezone.utc).isoformat(),'design':'Post hoc exploratory reference sensitivity after observing seed20261110 early deviation; not preregistered confirmation','seeds':SEEDS,'windows_ns':WINDOWS,'references':'all 38 NMR models, no best-model selection','fit_selections':{'all_ca':'positions1-20','interior_ca':'positions3-18, independently fitted'},'distance_mae':'All CA pairs separated by >=3 residues; rigid-fit-free','statistical_unit':'trajectory; NMR models are reference choices, not sample size','input_sha256':hashes,'script_sha256':sha(Path(__file__))}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    refs=ca_models(refpath,fasta_sequence(ROOT/'data/prepared/1L2Y.fasta'))
    assert len(refs)==38
    rows=[]; checks=[]
    i,j=np.triu_indices(20,k=3)
    for seed in SEEDS:
        times,xyz,ids=read_gro_trajectory(source/f'seed_{seed}/protein_whole.gro')
        validate_times(times,11)
        ca=xyz[:,[k for k,(_,_,name) in enumerate(ids) if name=='CA']]
        assert ca.shape==(1101,20,3)
        td=np.linalg.norm(ca[:,i]-ca[:,j],axis=-1)
        for number,ref,_ in refs:
            aligned=fitted_coordinates(ca,ref)
            rms=np.sqrt(np.mean(np.sum((aligned-ref)**2,axis=-1),axis=-1))
            core=fitted_coordinates(ca[:,2:18],ref[2:18])
            crms=np.sqrt(np.mean(np.sum((core-ref[2:18])**2,axis=-1),axis=-1))
            mae=np.mean(abs(td-np.linalg.norm(ref[i]-ref[j],axis=-1)),axis=-1)
            # Independent implementation, each reference at 0, 1.5 and 11 ns.
            for k in (0,150,1100):
                svd=SVDSuperimposer();svd.set(ref,ca[k]);svd.run()
                error=abs(float(rms[k])-svd.get_rms())
                assert error<1e-10
                checks.append(error)
            for start,end in WINDOWS:
                mask=(times>=start*1000)&((times<=end*1000) if end==11 else (times<end*1000))
                rows.append({'seed':seed,'nmr_model':number,'window_ns':f'{start}-{end}','frames':int(mask.sum()),'mean_ca_rmsd_angstrom':float(rms[mask].mean()),'mean_interior_ca_rmsd_angstrom':float(crms[mask].mean()),'mean_nonlocal_distance_mae_angstrom':float(mae[mask].mean())})
    write_csv(out/'all_reference_window_results.csv',rows)
    summaries=[]; contrasts=[]
    metrics=('mean_ca_rmsd_angstrom','mean_interior_ca_rmsd_angstrom','mean_nonlocal_distance_mae_angstrom')
    for start,end in WINDOWS:
        window=f'{start}-{end}'
        for metric in metrics:
            byseed={seed:np.array([r[metric] for r in rows if r['seed']==seed and r['window_ns']==window]) for seed in SEEDS}
            for seed,values in byseed.items():
                summaries.append({'seed':seed,'window_ns':window,'metric':metric,'reference_min':float(values.min()),'reference_median':float(np.median(values)),'reference_max':float(values.max())})
            contrasts.append({'window_ns':window,'metric':metric,'references_first_seed_exceeds_both_others':int(np.sum((byseed[SEEDS[0]]>byseed[SEEDS[1]])&(byseed[SEEDS[0]]>byseed[SEEDS[2]]))),'n_reference_choices':38,'first_minus_second_min':float((byseed[SEEDS[0]]-byseed[SEEDS[1]]).min()),'first_minus_second_max':float((byseed[SEEDS[0]]-byseed[SEEDS[1]]).max()),'first_minus_third_min':float((byseed[SEEDS[0]]-byseed[SEEDS[2]]).min()),'first_minus_third_max':float((byseed[SEEDS[0]]-byseed[SEEDS[2]]).max())})
    write_csv(out/'reference_sensitivity_summary.csv',summaries)
    write_csv(out/'paired_reference_contrasts.csv',contrasts)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    result={'status':'complete','independent_svd_checks':len(checks),'max_svd_difference_angstrom':max(checks),'contrasts':contrasts,'limitations':['Reference-choice ranges are not confidence intervals.','References are an experimental structural ensemble, not a Boltzmann sample or independent replicates.','Post hoc windows diagnose existing observations; they do not validate a mechanism or transition rate.','CA geometry does not fully measure sidechain chemistry.']}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for ax,metric,label in zip(axs,metrics[:2],['All 20 CA independently fitted','Positions 3-18 independently fitted']):
        for offset,seed in zip((-.2,0,.2),SEEDS):
            chosen=[r for r in summaries if r['seed']==seed and r['metric']==metric and r['window_ns'] in ('1-3','8-11')]
            med=np.array([r['reference_median'] for r in chosen]);lo=np.array([r['reference_min'] for r in chosen]);hi=np.array([r['reference_max'] for r in chosen])
            ax.errorbar(np.arange(2)+offset,med,yerr=[med-lo,hi-med],fmt='o',capsize=4,label=str(seed))
        ax.set(xticks=[0,1],xticklabels=['1-3 ns (post hoc)','8-11 ns (original)'],ylabel='Mean CA RMSD (Å)',title=label)
        ax.legend(fontsize=8)
    fig.suptitle('38 NMR reference choices: median and full range, not confidence intervals',fontsize=11)
    fig.tight_layout();fig.savefig(out/'reference_sensitivity.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
