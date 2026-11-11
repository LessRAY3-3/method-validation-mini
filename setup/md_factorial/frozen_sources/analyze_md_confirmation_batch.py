"""Frozen new-seed MD analysis: validate complete runs then report all endpoints.

No dynamics or model inference. Original batch is never modified. All new
trajectories are analyzed independently of their observed outcomes.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from analyze_md_replicates import (discover_runs, read_gro_trajectory,
    fitted_coordinates, validate_times, window_selection, sha)
from prediction_utils import ca_models, fasta_sequence

ROOT = Path(__file__).resolve().parents[1]
NEW_SEEDS = (20261113, 20261114, 20261115)
WINDOWS = ((0, 11), (1, 3), (8, 11))
CUTOFFS = (7, 8, 9)
FACTORS = (1.1, 1.2, 1.3)
PAIRS = ((3, 19), (12, 18))


def episodes(times, above):
    times = np.asarray(times); above = np.asarray(above, bool)
    if len(times) != len(above) or (len(times)>1 and not np.all(np.diff(times)==10)):
        raise ValueError('Expected matching 10 ps time grid')
    edges = np.diff(np.r_[False, above, False].astype(int))
    return [(int(times[a]), int(times[b-1]), int(b-a))
            for a,b in zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1))]


def selected_occupancy(reference_distance, actual_distances, cutoff, factor):
    if reference_distance > cutoff:
        return 'excluded', None
    return 'included', float(np.mean(np.asarray(actual_distances) <= factor*reference_distance))


def write_csv(path, rows, fields=None):
    with path.open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate_command(input_root, output, seeds, gmx):
    return [sys.executable, str(ROOT/'scripts/analyze_md_replicates.py'),
            '--input-root', str(input_root), '--output-dir', str(output),
            '--seeds', *map(str,seeds), '--gmx', gmx, '--total-ns', '11', '--keep-coordinates']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs=3, default=NEW_SEEDS)
    parser.add_argument('--gmx', default='gmx')
    args=parser.parse_args()
    seeds=tuple(args.seeds)
    # Require completed, internally consistent runs before creating output.
    runs=discover_runs(args.input_root.resolve(),11,seeds=seeds)
    output=args.output_dir.resolve()
    if output.exists():raise ValueError('Output exists; refusing to overwrite scientific evidence')
    sources=[ROOT/'data/raw/1L2Y.pdb',ROOT/'data/prepared/1L2Y.fasta',ROOT/'results/md/md.tpr']
    for _,directory,provenance in runs:
        for rel,want in provenance['input_sha256'].items():
            if sha(ROOT/rel)!=want:raise ValueError(f'Original input changed: {rel}')
        sources.extend(directory/name for name in ('md.xtc','md.tpr','provenance.json','md.done.json'))
    hashes={str(p):sha(p) for p in sources}
    scripts=['analyze_md_confirmation_batch.py','analyze_md_replicates.py','prediction_utils.py']
    output.mkdir(parents=True)
    declaration={'analysis_started_utc':datetime.now(timezone.utc).isoformat(),
        'classification':'New velocity-seed batch; analysis rules frozen before new dynamics by external experiment protocol' if seeds==NEW_SEEDS else 'Method verification using explicitly supplied seeds; not new-seed confirmation',
        'seeds':seeds,'windows_ns':WINDOWS,'cutoffs_A':CUTOFFS,'retention_factors':FACTORS,
        'distance_thresholds_A':[5,6,7], 'input_sha256':hashes,
        'source_sha256':{name:sha(ROOT/'scripts'/name) for name in scripts},
        'note':'This analysis-time declaration is not the run-start preregistration; see external batch protocol. n=3 trajectories; references and frames are not independent replicates.'}
    (output/'analysis_manifest.json').write_text(json.dumps(declaration,indent=2)+'\n')
    command=aggregate_command(args.input_root.resolve(),output/'aggregate_analysis',seeds,args.gmx)
    with (output/'aggregate_console.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    refs=ca_models(ROOT/'data/raw/1L2Y.pdb',fasta_sequence(ROOT/'data/prepared/1L2Y.fasta'))
    if len(refs)!=38:raise ValueError('Expected all38 NMR reference models')
    window_rows=[]; threshold_rows=[]; episode_rows=[]; reference_rows=[]; q_rows=[]; contact_rows=[]
    check_errors=[]
    i,j=np.triu_indices(20,k=3)
    for seed,_,_ in runs:
        dest=output/'aggregate_analysis'/f'seed_{seed}'
        times,xyz,ids=read_gro_trajectory(dest/'protein_whole.gro');validate_times(times,11)
        indices=[k for k,(_,_,name) in enumerate(ids) if name=='CA']
        if [ids[k][0] for k in indices]!=list(range(1,21)):raise ValueError('CA residue order mismatch')
        ca=xyz[:,indices]
        distances=np.linalg.norm(ca[:,i]-ca[:,j],axis=-1)
        with (dest/'timeseries.csv').open() as f:series=list(csv.DictReader(f))
        validate_times([float(r['time_ps']) for r in series],11)
        metrics={key:np.array([float(row[key]) for row in series]) for key in series[0] if key!='time_ps'}
        for a,b in WINDOWS:
            mask=window_selection(times,a,b,11);window=f'{a}-{b}'
            window_rows.append({'seed':seed,'window_ns':window,'frames':int(mask.sum()),**{key:float(values[mask].mean()) for key,values in metrics.items()}})
            for threshold in (5,6,7):
                above=metrics['sidechain_min_3_19_angstrom'][mask]>threshold
                spans=episodes(times[mask],above)
                threshold_rows.append({'seed':seed,'window_ns':window,'threshold_A':threshold,'sample_fraction_above':float(above.mean()),'n_samples_above':int(above.sum()),'n_sampled_episodes':len(spans),'longest_sampled_span_ps':max((end-start for start,end,n in spans),default=0)})
                episode_rows.extend({'seed':seed,'window_ns':window,'threshold_A':threshold,'first_sample_ps':start,'last_sample_ps':end,'n_samples':n,'sampled_span_ps':end-start} for start,end,n in spans)
        native=np.linalg.norm(refs[0][1][i]-refs[0][1][j],axis=-1)
        for cutoff in CUTOFFS:
            selected=native<=cutoff
            for factor in FACTORS:
                q=np.mean(distances[:,selected]<=factor*native[selected],axis=1)
                for a,b in WINDOWS:
                    mask=window_selection(times,a,b,11);window=f'{a}-{b}'
                    q_rows.append({'seed':seed,'window_ns':window,'native_cutoff_A':cutoff,'retention_factor':factor,'native_pair_count':int(selected.sum()),'mean_q':float(q[mask].mean())})
                    for first,second in PAIRS:
                        index=int(np.flatnonzero((i==first-1)&(j==second-1))[0])
                        status,value=selected_occupancy(native[index],distances[mask,index],cutoff,factor)
                        contact_rows.append({'seed':seed,'window_ns':window,'native_cutoff_A':cutoff,'retention_factor':factor,'residue_i':first,'residue_j':second,'reference_distance_A':float(native[index]),'selection_status':status,'occupancy':value})
        for number,ref,_ in refs:
            aligned=fitted_coordinates(ca,ref)
            rmsd=np.sqrt(np.mean(np.sum((aligned-ref)**2,axis=-1),axis=-1))
            internal=fitted_coordinates(ca[:,2:18],ref[2:18])
            irmsd=np.sqrt(np.mean(np.sum((internal-ref[2:18])**2,axis=-1),axis=-1))
            mae=np.mean(abs(distances-np.linalg.norm(ref[i]-ref[j],axis=-1)),axis=-1)
            for k in (0,150,1100):
                svd=SVDSuperimposer();svd.set(ref,ca[k]);svd.run()
                error=abs(float(rmsd[k])-svd.get_rms())
                if error>1e-10:raise ValueError('Independent SVD mismatch')
                check_errors.append(error)
            for a,b in WINDOWS:
                mask=window_selection(times,a,b,11)
                reference_rows.append({'seed':seed,'window_ns':f'{a}-{b}','nmr_model':number,'mean_ca_rmsd_angstrom':float(rmsd[mask].mean()),'mean_interior_ca_rmsd_angstrom':float(irmsd[mask].mean()),'mean_nonlocal_distance_mae_angstrom':float(mae[mask].mean())})
    for name,rows in [('per_trajectory_windows.csv',window_rows),('threshold_sensitivity.csv',threshold_rows),('all_reference_window_results.csv',reference_rows),('all_q_windows.csv',q_rows),('selected_contact_occupancies.csv',contact_rows)]:write_csv(output/name,rows)
    write_csv(output/'threshold_episodes.csv',episode_rows,['seed','window_ns','threshold_A','first_sample_ps','last_sample_ps','n_samples','sampled_span_ps'])
    if any(sha(Path(path))!=value for path,value in hashes.items()):raise ValueError('Input changed during analysis')
    result={'status':'three_completed_confirmation_replicates_verified' if seeds==NEW_SEEDS else 'original_seed_method_verification_passed','seeds':seeds,'n_trajectories':3,'frames_per_trajectory':1101,'total_ns_each':11,'independent_svd_checks':len(check_errors),'maximum_svd_difference_A':max(check_errors),'limitations':['Different initial velocities from one shared minimized configuration.','First batch and new batch must be reported separately before descriptive combination.','References, windows and correlated frames do not increase n.','Threshold crossing sample spans are not exact state lifetimes.','No frame-level p-values, folding rates or convergence claims.'],'output_sha256':{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'output':str(output)}))

if __name__=='__main__':main()
