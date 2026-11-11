"""Geometry-only diagnosis of the nine archived MSA ablation predictions.

No simulation, model inference or network calls. 3-18 denotes a positional
interior subset, not a chemically defined hydrophobic core.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.vectors import Vector, calc_dihedral
from prediction_utils import AA, ca_models, fitted_ca_rmsd
from prediction_provenance import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/prediction_msa_diagnosis/structure'
SEQ = 'NLYIQWLKDGGPSSGRPPPS'
CONDITIONS = ['query_only', 'deduplicated_saved_msa', 'raw_saved_msa']
SEEDS = [42, 2026, 20261110]
INTERIOR = np.arange(2, 18)
TERMINAL = np.array([0, 1, 18, 19])
PAIRS = [(6, 12), (6, 18), (3, 19)]


def distance_map(x):
    x = np.asarray(x, dtype=float)
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def dihedral_deg(a, b, c, d):
    """Signed torsion using projections onto the plane normal to b->c."""
    b0 = np.asarray(a, dtype=float) - b
    b1 = np.asarray(c, dtype=float) - b
    b2 = np.asarray(d, dtype=float) - c
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-10:
        raise ValueError('Undefined torsion for collinear atoms')
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def angular_delta(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.) % 360. - 180.


def load_models(path):
    structure = PDBParser(QUIET=True).get_structure('structure', path)
    models = []
    for model in structure:
        residues = [r for r in model.get_residues() if r.id[0] == ' ']
        if ''.join(AA.get(r.resname, 'X') for r in residues) != SEQ:
            raise ValueError(f'Sequence mismatch: {path}')
        if [r.id[1] for r in residues] != list(range(1, 21)) or len({r.parent.id for r in residues}) != 1:
            raise ValueError('Expected one chain numbered 1-20')
        for r in residues:
            if any(name not in r for name in ['N', 'CA', 'C', 'O']):
                raise ValueError('Missing backbone atom')
            for a in r:
                if a.is_disordered() or not np.isfinite(a.coord).all():
                    raise ValueError('Disordered/nonfinite atom')
        models.append(residues)
    return models


def ca(residues):
    return np.array([r['CA'].coord for r in residues], dtype=float)


def fitted_errors(mobile_res, reference_res, indices):
    """Bio.PDB independent SVD implementation with declared fit atom selection."""
    fitter = Superimposer()
    fitter.set_atoms([reference_res[i]['CA'] for i in indices], [mobile_res[i]['CA'] for i in indices])
    rot, trans = fitter.rotran
    return np.linalg.norm(ca(mobile_res) @ rot + trans - ca(reference_res), axis=1)


def torsions(res):
    rows = []
    max_difference = 0.
    for i, r in enumerate(res):
        row = {'residue': i+1, 'aa': SEQ[i], 'phi_deg': None, 'psi_deg': None}
        quads = {}
        if i:
            quads['phi_deg'] = [res[i-1]['C'], r['N'], r['CA'], r['C']]
        if i+1 < len(res):
            quads['psi_deg'] = [r['N'], r['CA'], r['C'], res[i+1]['N']]
        for key, atoms in quads.items():
            xyz = [np.array(a.coord, dtype=float) for a in atoms]
            value = dihedral_deg(*xyz)
            independent = float(np.degrees(calc_dihedral(*[Vector(x) for x in xyz])))
            max_difference = max(max_difference, abs(float(angular_delta(value, independent))))
            row[key] = value
        rows.append(row)
    if max_difference > 1e-6:
        raise AssertionError('Torsion sign/convention mismatch with Bio.PDB')
    return rows, max_difference


def sidechain_min(res, i, j):
    backbone = {'N', 'CA', 'C', 'O', 'OXT'}
    atomsets = [[a for a in res[k-1] if a.element not in {'H', 'D'} and a.name not in backbone] for k in [i, j]]
    if any(not atoms for atoms in atomsets):
        raise ValueError('Empty side chain')
    distances = np.array([[np.linalg.norm(a.coord-b.coord) for b in atomsets[1]] for a in atomsets[0]])
    a, b = np.unravel_index(np.argmin(distances), distances.shape)
    return float(distances[a,b]), atomsets[0][a].name, atomsets[1][b].name


def geometry(res):
    cn = [float(np.linalg.norm(res[i]['C'].coord-res[i+1]['N'].coord)) for i in range(19)]
    adjacent_ca = np.diag(distance_map(ca(res)), 1)
    # Gross clash screen only: nonadjacent residues, no chemistry-based validation.
    nonadjacent = []
    for i in range(20):
        a=np.array([a.coord for a in res[i] if a.element not in {'H', 'D'}])
        for j in range(i+2,20):
            b=np.array([a.coord for a in res[j] if a.element not in {'H','D'}])
            d=float(np.linalg.norm(a[:,None,:]-b[None,:,:],axis=-1).min())
            if d < 1.5:
                nonadjacent.append({'residues':[i+1,j+1],'minimum_heavy_distance_angstrom':d})
    return {'peptide_cn_min_angstrom':min(cn),'peptide_cn_max_angstrom':max(cn),
            'peptide_cn_outside_1_15_to_1_55': [i+1 for i,v in enumerate(cn) if not 1.15<=v<=1.55],
            'adjacent_ca_min_angstrom':float(adjacent_ca.min()),'adjacent_ca_max_angstrom':float(adjacent_ca.max()),
            'adjacent_ca_outside_3_5_to_4_1':[i+1 for i,v in enumerate(adjacent_ca) if not 3.5<=v<=4.1],
            'gross_nonadjacent_heavy_clashes_below_1_5':nonadjacent}


def write_csv(name, rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ref_path=ROOT/'data/raw/1L2Y.pdb'
    references=load_models(ref_path)
    assert len(references)==38
    ref_maps=np.array([distance_map(ca(r)) for r in references])
    ref_torsions=[torsions(r)[0] for r in references]
    ref_contacts={f'{i}-{j}':np.array([sidechain_min(r,i,j)[0] for r in references]) for i,j in PAIRS}
    upper=np.triu_indices(20,1)
    interior_upper=np.triu_indices(16,1)
    summaries=[];errors_rows=[];dihedral_rows=[];contact_rows=[];dist_rows=[];ref_contact_rows=[];geometry_rows=[]
    per_reference=[]
    cases={};max_rmsd_disagreement=0.;max_torsion_disagreement=0.
    source_hashes={str(ref_path.relative_to(ROOT)):sha256(ref_path)}
    for pair, vals in ref_contacts.items():
        for model,value in enumerate(vals,1):ref_contact_rows.append({'pair':pair,'nmr_model':model,'minimum_sidechain_heavy_distance_angstrom':value})
    for condition in CONDITIONS:
        for seed in SEEDS:
            directory=ROOT/'results/prediction_ablation/runs'/condition/f'seed_{seed}'
            paths=list(directory.glob('*unrelaxed*.pdb'));assert len(paths)==1
            path=paths[0];source_hashes[str(path.relative_to(ROOT))]=sha256(path)
            manifest=json.loads((directory/'manifest.json').read_text())
            assert source_hashes[str(path.relative_to(ROOT))]==manifest['output_sha256'][path.name]
            residues=load_models(path);assert len(residues)==1
            residues=residues[0];coords=ca(residues)
            independent_coords=ca_models(path,SEQ)[0][1]
            assert np.allclose(coords,independent_coords,atol=1e-6,rtol=0)
            global_errors=np.array([fitted_errors(residues,r,np.arange(20)) for r in references])
            subset_errors=np.array([fitted_errors(residues,r,INTERIOR) for r in references])
            full_rmsd=np.sqrt((global_errors**2).mean(1))
            for index,r in enumerate(references):
                independent=fitted_ca_rmsd(independent_coords,ca(r))
                max_rmsd_disagreement=max(max_rmsd_disagreement,abs(full_rmsd[index]-independent))
            dm=distance_map(coords);absolute_map_error=np.abs(ref_maps-dm[None,:,:])
            dm_mae=absolute_map_error[:,upper[0],upper[1]].mean(1)
            dm_interior=absolute_map_error[:,2:18,2:18][:,interior_upper[0],interior_upper[1]].mean(1)
            for k in range(38):
                per_reference.append({'condition':condition,'seed':seed,'nmr_model':k+1,
                                      'full_fit_full_rmsd_angstrom':float(full_rmsd[k]),
                                      'interior_fit_interior_rmsd_angstrom':float(np.sqrt((subset_errors[k,INTERIOR]**2).mean())),
                                      'ca_distance_mae_angstrom':float(dm_mae[k]),
                                      'interior_ca_distance_mae_angstrom':float(dm_interior[k])})
            row={'condition':condition,'seed':seed,'model1_biopdb_ca_rmsd_angstrom':float(full_rmsd[0]),
                 'median_nmr_full_fit_full_rmsd_angstrom':float(np.median(full_rmsd)),
                 'median_nmr_full_fit_interior_3_18_rmsd_angstrom':float(np.median(np.sqrt((global_errors[:,INTERIOR]**2).mean(1)))),
                 'median_nmr_interior_fit_interior_3_18_rmsd_angstrom':float(np.median(np.sqrt((subset_errors[:,INTERIOR]**2).mean(1)))),
                 'median_nmr_interior_fit_terminal_rmsd_angstrom':float(np.median(np.sqrt((subset_errors[:,TERMINAL]**2).mean(1)))),
                 'median_nmr_ca_distance_mae_angstrom':float(np.median(dm_mae)),
                 'median_nmr_interior_ca_distance_mae_angstrom':float(np.median(dm_interior))}
            summaries.append(row)
            for k in range(20):
                errors_rows.append({'condition':condition,'seed':seed,'residue':k+1,'aa':SEQ[k],
                                   'median_nmr_full_fit_displacement_angstrom':float(np.median(global_errors[:,k])),
                                   'median_nmr_interior_fit_displacement_angstrom':float(np.median(subset_errors[:,k])),
                                   'median_nmr_distance_profile_mae_angstrom':float(np.median(absolute_map_error[:,k,:].sum(1)/19))})
            for i,j in zip(*upper):
                dist_rows.append({'condition':condition,'seed':seed,'residue_i':i+1,'residue_j':j+1,'predicted_ca_distance_angstrom':float(dm[i,j]),'nmr_median_ca_distance_angstrom':float(np.median(ref_maps[:,i,j])),'median_nmr_absolute_error_angstrom':float(np.median(absolute_map_error[:,i,j]))})
            ts,disagreement=torsions(residues);max_torsion_disagreement=max(max_torsion_disagreement,disagreement)
            for t in ts:
                for key in ['phi_deg','psi_deg']:
                    if t[key] is None:continue
                    values=np.array([r[t['residue']-1][key] for r in ref_torsions])
                    mean=float(np.degrees(np.angle(np.exp(1j*np.radians(values)).mean())))
                    dihedral_rows.append({'condition':condition,'seed':seed,'residue':t['residue'],'aa':t['aa'],'torsion':key[:-4],
                                          'predicted_degrees':t[key],'nmr_circular_mean_degrees':mean,'nmr_max_angular_deviation_from_mean_degrees':float(np.abs(angular_delta(values,mean)).max()),
                                          'median_absolute_nmr_angular_difference_degrees':float(np.median(np.abs(angular_delta(t[key],values))))})
            for i,j in PAIRS:
                distance,a,b=sidechain_min(residues,i,j);values=ref_contacts[f'{i}-{j}']
                contact_rows.append({'condition':condition,'seed':seed,'pair':f'{i}-{j}','sidechain_atom_i':a,'sidechain_atom_j':b,'minimum_sidechain_heavy_distance_angstrom':distance,
                                     'nmr_min_angstrom':float(values.min()),'nmr_median_angstrom':float(np.median(values)),'nmr_max_angstrom':float(values.max()),
                                     'outside_nmr_observed_range':bool(distance<values.min() or distance>values.max())})
            geo=geometry(residues);geometry_rows.append({'condition':condition,'seed':seed,**geo})
            cases[condition,seed]={'errors':np.median(global_errors,0),'subset_errors':np.median(subset_errors,0),'distance_error':np.median(absolute_map_error,0)}
    assert max_rmsd_disagreement<1e-5
    for name,rows in [('per_reference.csv',per_reference),('summary.csv',summaries),('per_residue.csv',errors_rows),('distance_pairs.csv',dist_rows),('dihedrals.csv',dihedral_rows),('sidechain_contacts.csv',contact_rows),('nmr_sidechain_contacts.csv',ref_contact_rows)]:write_csv(name,rows)
    result={'source_sha256':source_hashes,'model_count':9,'reference_model_count':38,
            'independent_validation':{'biopdb_vs_numpy_kabsch_max_difference_angstrom':max_rmsd_disagreement,'dihedral_vs_biopdb_max_circular_difference_degrees':max_torsion_disagreement,'all_sequences_equal':SEQ},
            'fit_definitions':{'global':'All 20 CA atoms','interior':'CA at positions 3-18 inclusive; positional subset, not a hydrophobic-core definition'},
            'distance_map_definition':'Absolute CA-CA distance error vs each NMR model; upper triangle excluding diagonal, then median of per-model MAE over 38 models',
            'torsion_definitions':{'phi':'C(i-1), N(i), CA(i), C(i)','psi':'N(i), CA(i), C(i), N(i+1)'},
            'geometry_screen_caveat':'Broad peptide connectivity and severe nonadjacent heavy-atom overlap screens, not MolProbity or energetic validation. These are unrelaxed predictions.',
            'rows':summaries,'geometry':geometry_rows,
            'limitations':'Post hoc descriptive diagnosis of one outlying archived prediction; NMR models are not independent replicates. Geometry cannot establish the internal stochastic/MSA mechanism or thermodynamic state. Side-chain contacts are minimum distances, not binding interactions.'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    labels={'query_only':'query','raw_saved_msa':'raw','deduplicated_saved_msa':'dedup'}
    focus=[('deduplicated_saved_msa',20261110),('deduplicated_saved_msa',42),('deduplicated_saved_msa',2026),('query_only',20261110),('raw_saved_msa',20261110)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for key in focus:
        label=f'{labels[key[0]]}, seed {key[1]}'
        axes[0].plot(range(1,21),cases[key]['errors'],marker='.',label=label)
        axes[1].plot(range(1,21),cases[key]['subset_errors'],marker='.',label=label)
    for ax in axes:
        ax.set(xlabel='Residue position',ylabel='Median NMR CA displacement (Å)',xticks=[1,3,6,9,12,15,18,20]);ax.axvspan(.5,2.5,color='gray',alpha=.1);ax.axvspan(18.5,20.5,color='gray',alpha=.1)
    axes[0].set_title('All 20 CA atoms fitted');axes[1].set_title('Only positions 3-18 fitted');axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'alignment_sensitivity.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),constrained_layout=True)
    keys=[('deduplicated_saved_msa',20261110),('query_only',20261110),('raw_saved_msa',20261110)]
    vmax=max(cases[k]['distance_error'].max() for k in keys)
    for ax,key in zip(axes,keys):
        im=ax.imshow(cases[key]['distance_error'],vmin=0,vmax=vmax,origin='lower',extent=[.5,20.5,.5,20.5],cmap='magma');ax.set(title=f'{labels[key[0]]}, seed {key[1]}',xlabel='Residue position',ylabel='Residue position',xticks=[1,5,10,15,20],yticks=[1,5,10,15,20])
    fig.colorbar(im,ax=axes,label='Median absolute CA distance error (Å)',shrink=.8)
    fig.savefig(OUT/'distance_map_errors.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    for ax,torsion in zip(axes,['phi','psi']):
        for key in focus:
            rows=[r for r in dihedral_rows if (r['condition'],r['seed'])==key and r['torsion']==torsion]
            ax.plot([r['residue'] for r in rows],[r['median_absolute_nmr_angular_difference_degrees'] for r in rows],marker='.',label=f'{labels[key[0]]} {key[1]}')
        ax.set(title=f'{torsion}: four-atom torsion difference',xlabel='Residue position',ylabel='Median absolute angular difference (degrees)',xticks=[1,5,10,15,20])
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'torsion_differences.png',dpi=180);plt.close(fig)
    print(json.dumps({'independent_validation':result['independent_validation'],'rows':summaries,'geometry':geometry_rows},indent=2))


if __name__=='__main__':
    main()
