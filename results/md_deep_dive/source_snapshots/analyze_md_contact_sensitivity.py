"""Post-hoc reference-cutoff and contact-retention-factor sensitivity; no dynamics."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_md_replicates import read_gro_trajectory, validate_times, window_selection
from analyze_md_local_geometry import pair_geometry, SEEDS, WINDOWS
from prediction_utils import ca_models, fasta_sequence
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/md_deep_dive/contact_sensitivity'
CUTOFFS=(7,8,9)
FACTORS=(1.1,1.2,1.3)
PAIRS=((3,19),(12,18))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():raise ValueError('Output exists; refusing overwrite')
    OUT.mkdir(parents=True)
    paths=[ROOT/'data/raw/1L2Y.pdb',*[ROOT/f'results/md_replicates_gpu/aggregate_analysis_audited/seed_{s}/protein_whole.gro' for s in SEEDS]]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    plan={'declared_utc':datetime.now(timezone.utc).isoformat(),'classification':'Exploratory sensitivity extension after examining first-seed early deviation and cutoff-only results; not independent confirmation','seed_order':SEEDS,'windows_ns':WINDOWS,'reference_cutoffs_A':CUTOFFS,'retention_factors':FACTORS,'selected_CA_pairs':PAIRS,'excluded_pair_rule':'Mark excluded and occupancy=null if reference distance exceeds cutoff; never encode excluded as0','formula':'Reference CA pairs separation>=3 and distance<=cutoff; contact retained when trajectory distance<=factor*reference distance','input_sha256':hashes,'script_sha256':sha(__file__)}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    ref=ca_models(ROOT/'data/raw/1L2Y.pdb',fasta_sequence(ROOT/'data/prepared/1L2Y.fasta'))[0][1]
    rows=[];selected=[];checks=[]
    for seed in SEEDS:
        p=ROOT/f'results/md_replicates_gpu/aggregate_analysis_audited/seed_{seed}/protein_whole.gro'
        times,xyz,ids=read_gro_trajectory(p);validate_times(times,11)
        ca=xyz[:,[k for k,x in enumerate(ids) if x[2]=='CA']]
        i,j,native,distances,_=pair_geometry(ca,ref)
        for cutoff in CUTOFFS:
            keep=native<=cutoff
            for factor in FACTORS:
                q=np.mean(distances[:,keep]<=factor*native[keep],axis=1)
                for start,end in WINDOWS:
                    mask=window_selection(times,start,end,11);window=f'{start}-{end}'
                    rows.append({'seed':seed,'window_ns':window,'frames':int(mask.sum()),'native_cutoff_A':cutoff,'retention_factor':factor,'native_pair_count':int(keep.sum()),'mean_q':float(q[mask].mean())})
                    for a,b in PAIRS:
                        k=int(np.flatnonzero((i==a-1)&(j==b-1))[0]);included=bool(native[k]<=cutoff)
                        selected.append({'seed':seed,'window_ns':window,'native_cutoff_A':cutoff,'retention_factor':factor,'residue_i':a,'residue_j':b,'reference_distance_A':float(native[k]),'selection_status':'included' if included else 'excluded','occupancy':float(np.mean(distances[mask,k]<=factor*native[k])) if included else None})
                # Independent scalar loop at one frame per seed and all nine definitions.
                values=[]
                for a in range(20):
                    for b in range(a+3,20):
                        d=sum(float(ref[a,k]-ref[b,k])**2 for k in range(3))**.5
                        if d<=cutoff:
                            actual=sum(float(ca[150,a,k]-ca[150,b,k])**2 for k in range(3))**.5
                            values.append(actual<=factor*d)
                error=abs(float(np.mean(values))-float(q[150]));assert error<1e-12
                checks.append({'seed':seed,'frame':150,'cutoff_A':cutoff,'factor':factor,'absolute_q_error':error})
    for name,data in [('all_q_windows.csv',rows),('selected_contact_occupancies.csv',selected)]:
        with (OUT/name).open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    comparisons=[]
    for cutoff in CUTOFFS:
        for factor in FACTORS:
            def value(seed,window):return next(r['mean_q'] for r in rows if r['seed']==seed and r['window_ns']==window and r['native_cutoff_A']==cutoff and r['retention_factor']==factor)
            early=[value(s,'1-3') for s in SEEDS];late=[value(s,'8-11') for s in SEEDS]
            comparisons.append({'native_cutoff_A':cutoff,'retention_factor':factor,'early_q_seed_order':early,'late_q_seed_order':late,'first_seed_strictly_lowest_early':early[0]<min(early[1:]),'late_minus_early_seed_order':[b-a for a,b in zip(early,late)]})
    selected_deltas=[]
    for seed in SEEDS:
        for cutoff in CUTOFFS:
            for factor in FACTORS:
                for a,b in PAIRS:
                    group=[r for r in selected if r['seed']==seed and r['native_cutoff_A']==cutoff and r['retention_factor']==factor and r['residue_i']==a and r['residue_j']==b]
                    early=next(r for r in group if r['window_ns']=='1-3');late=next(r for r in group if r['window_ns']=='8-11')
                    selected_deltas.append({'seed':seed,'cutoff_A':cutoff,'factor':factor,'pair':[a,b],'selection_status':early['selection_status'],'early':early['occupancy'],'late':late['occupancy'],'late_minus_early':late['occupancy']-early['occupancy'] if early['occupancy'] is not None else None})
    for path,want in hashes.items():assert sha(ROOT/path)==want
    summary={'status':'completed_posthoc_sensitivity','seed_order':SEEDS,'comparisons':comparisons,'first_seed_lowest_early_combinations':sum(x['first_seed_strictly_lowest_early'] for x in comparisons),'first_seed_q_early_to_late_positive_combinations':sum(x['late_minus_early_seed_order'][0]>0 for x in comparisons),'selected_pair_deltas':selected_deltas,'independent_scalar_checks':checks,'limitations':['Definitions applied to same three correlated trajectories; not9 independent experiments','Absolute occupancy depends on thresholds; rank/direction stability is descriptive','Excludes P12-P18 at7A because reference distance7.540A exceeds cutoff','Selected pairs/windows were chosen after inspecting the event; no causal or equilibrium inference'],'plan_sha256':sha(OUT/'plan.json')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'first_seed_lowest_early_combinations':summary['first_seed_lowest_early_combinations'],'first_seed_q_early_to_late_positive_combinations':summary['first_seed_q_early_to_late_positive_combinations'],'output':str(OUT)},indent=2))


if __name__=='__main__':main()
