"""Validate and summarize the prespecified paired-seed docking experiment."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from spyrmsd import rmsd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/docking_benchmark'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(successes, count):
    z=1.959963984540054
    p=successes/count
    center=(p+z*z/(2*count))/(1+z*z/count)
    half=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [max(0.,center-half),min(1.,center+half)]


def contacts(coords, receptor):
    return {label for label,positions in receptor.items() if np.any(np.linalg.norm(positions[:,None,:]-coords[None,:,:],axis=2)<=4.0)}


def receptor_atoms():
    atoms={}
    for line in (ROOT/'data/raw/1iep_receptorH.pdb').read_text().splitlines():
        if not line.startswith('ATOM') or line[76:78].strip() in {'H','D'}:
            continue
        label=f'{line[21]}:{line[17:20].strip()}{line[22:26].strip()}{line[26].strip()}'
        atoms.setdefault(label,[]).append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
    return {k:np.array(v) for k,v in atoms.items()}


def csv_write(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='Study directory; relative paths resolve from the repository root')
    args=parser.parse_args()
    if args.output_dir is not None:
        OUT = args.output_dir.resolve() if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    declaration=json.loads((OUT/'protocol.json').read_text());protocol=declaration['protocol']
    for file,value in protocol['inputs_sha256'].items():
        assert sha(ROOT/file)==value, f'Input changed: {file}'
    ref=Chem.RemoveHs(Chem.SDMolSupplier(str(ROOT/'data/raw/1iep_ligand.sdf'),removeHs=False)[0])
    ref_xyz=ref.GetConformer().GetPositions();ref_z=np.array([a.GetAtomicNum() for a in ref.GetAtoms()]);ref_adj=Chem.GetAdjacencyMatrix(ref)
    receptor=receptor_atoms();native=contacts(ref_xyz,receptor)
    assert native, 'No native contacts'
    all_rows=[];runs=[]
    for e in protocol['exhaustiveness']:
        for seed in protocol['paired_seeds']:
            folder=OUT/f'e{e}_s{seed}';record=json.loads((folder/'run.json').read_text())
            assert record['status']=='complete', f'Run incomplete: {folder}'
            assert record['started_utc']>=declaration['declared_before_runs_utc'], 'Protocol must precede experiment'
            for name,value in record['output_sha256'].items():
                assert sha(folder/name)==value, f'Output changed: {folder/name}'
            scores=[float(x) for x in re.findall(r'REMARK VINA RESULT:\s+(-?[\d.]+)',(folder/'poses.pdbqt').read_text())]
            molecules=list(Chem.SDMolSupplier(str(folder/'poses.sdf'),removeHs=False))
            assert len(scores)==len(molecules)>0
            pose_rows=[]
            for rank,m in enumerate(molecules,1):
                assert m is not None
                m=Chem.RemoveHs(m);xyz=m.GetConformer().GetPositions();z=np.array([a.GetAtomicNum() for a in m.GetAtoms()]);adj=Chem.GetAdjacencyMatrix(m)
                assert m.GetNumAtoms()==ref.GetNumAtoms()
                mapping=ref.GetSubstructMatch(m);assert mapping
                ix=np.array(mapping)
                value=float(rmsd.symmrmsd(ref_xyz[ix],xyz,ref_z[ix],z,ref_adj[np.ix_(ix,ix)],adj,center=False,minimize=False))
                # Independent RDKit graph enumeration + NumPy geometry, all 15 runs / all saved poses.
                alternatives=ref.GetSubstructMatches(m,uniquify=False,maxMatches=10000)
                independent=min(float(np.sqrt(np.mean(np.sum((ref_xyz[list(mp)]-xyz)**2,axis=1)))) for mp in alternatives)
                assert abs(value-independent)<1e-8
                found=contacts(xyz,receptor);shared=native&found
                row={'exhaustiveness':e,'seed':seed,'rank':rank,'affinity_kcal_mol':scores[rank-1],
                     'crystal_rmsd_angstrom':value,'success_le_2A':value<=2.0,
                     'contact_recall':len(shared)/len(native),'contact_precision':len(shared)/len(found) if found else 0.,
                     'contact_jaccard':len(shared)/len(native|found),'contact_residue_count':len(found),
                     'contact_residues':';'.join(sorted(found))}
                pose_rows.append(row);all_rows.append(row)
            assert all(scores[i]<=scores[i+1] for i in range(len(scores)-1)), 'Saved score order not monotonic'
            top=pose_rows[0];closest=min(pose_rows,key=lambda r:r['crystal_rmsd_angstrom'])
            runs.append({'exhaustiveness':e,'seed':seed,'saved_poses':len(pose_rows),
                         'top1_score_kcal_mol':top['affinity_kcal_mol'],'top1_rmsd_angstrom':top['crystal_rmsd_angstrom'],
                         'best_saved_rmsd_angstrom':closest['crystal_rmsd_angstrom'],'closest_pose_rank':closest['rank'],
                         'top1_success':top['success_le_2A'],'any_saved_success':closest['success_le_2A'],
                         'top1_native_contact_recall':top['contact_recall'],'top1_contact_jaccard':top['contact_jaccard'],
                         'vina_elapsed_seconds':record['vina_elapsed_seconds']})
    groups=[]
    for e in protocol['exhaustiveness']:
        subset=[r for r in runs if r['exhaustiveness']==e];n=len(subset)
        top=sum(r['top1_success'] for r in subset);any_saved=sum(r['any_saved_success'] for r in subset)
        groups.append({'exhaustiveness':e,'n':n,'top1_successes':top,'top1_success_fraction':top/n,
                       'top1_wilson95':wilson(top,n),'any_saved_successes':any_saved,'any_saved_success_fraction':any_saved/n,
                       'any_saved_wilson95':wilson(any_saved,n),
                       'median_top1_rmsd_angstrom':float(np.median([r['top1_rmsd_angstrom'] for r in subset])),
                       'median_vina_elapsed_seconds':float(np.median([r['vina_elapsed_seconds'] for r in subset]))})
    csv_write('all_poses.csv',all_rows);csv_write('run_summary.csv',runs)
    result={'protocol_sha256':sha(OUT/'protocol.json'),'complete_runs':len(runs),'saved_poses':len(all_rows),
            'native_contact_residues':sorted(native),'native_contact_count':len(native),'groups':groups,
            'independent_rmsd_check':'All saved poses agree with RDKit enumeration + NumPy within 1e-8 angstrom',
            'contact_caveat':'Binary residue contacts use a geometric 4A threshold; no bond type, affinity or pharmacological inference. High recovery does not imply correct ligand orientation.',
            'limitations':[protocol['generalization_limit'],protocol['timing_limit'],
                          'n=5 seeds per effort; Wilson intervals are descriptive and do not model target/preparation uncertainty.',
                          'Energy-window filtered saved poses are not the complete conformational search history.']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.2))
    for seed in protocol['paired_seeds']:
        subset=[r for r in runs if r['seed']==seed]
        axs[0].plot([r['exhaustiveness'] for r in subset],[r['top1_rmsd_angstrom'] for r in subset],marker='o',label=str(seed),alpha=.8)
    axs[0].axhline(2,color='black',ls='--',lw=1);axs[0].set(xscale='log',xlabel='Exhaustiveness',ylabel='Top-1 crystal RMSD (Å)',xticks=protocol['exhaustiveness']);axs[0].set_xticklabels(protocol['exhaustiveness']);axs[0].legend(title='Paired seed',fontsize=7)
    for i,g in enumerate(groups):
        for offset,key,color in [(-.12,'top1','#147d86'),(.12,'any_saved','#ad6a27')]:
            p=g[f'{key}_success_fraction'];lo,hi=g[f'{key}_wilson95']
            axs[1].errorbar(i+offset,p,yerr=[[p-lo],[hi-p]],fmt='o',color=color,capsize=4,label=('Top-1' if key=='top1' else 'Any saved pose') if i==0 else None)
    axs[1].set(xticks=range(len(groups)),xticklabels=protocol['exhaustiveness'],xlabel='Exhaustiveness (5 seeds each)',ylabel='Success fraction; Wilson 95% interval',ylim=(-.05,1.08));axs[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'paired_seed_outcomes.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.2))
    for e in protocol['exhaustiveness']:
        sub=[r for r in all_rows if r['exhaustiveness']==e]
        axs[0].scatter([r['crystal_rmsd_angstrom'] for r in sub],[r['affinity_kcal_mol'] for r in sub],label=f'e={e}',s=26,alpha=.7)
        axs[1].scatter([r['crystal_rmsd_angstrom'] for r in sub],[r['contact_recall'] for r in sub],label=f'e={e}',s=26,alpha=.7)
    for ax in axs:ax.axvline(2,color='black',ls='--',lw=1);ax.set_xlabel('Crystal RMSD (Å)');ax.legend(fontsize=8)
    axs[0].set_ylabel('Vina score (kcal/mol)');axs[1].set_ylabel('Native residue contact recall (4 Å)')
    fig.tight_layout();fig.savefig(OUT/'score_contacts_vs_rmsd.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
