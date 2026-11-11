"""Independent, offline checks of committed inputs and docking results.

This checks artifacts, not whether a simulation or inference was rerun.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root=ROOT):
    checks = {}
    with (root / 'data/raw/SOURCES.tsv').open() as handle:
        sources = list(csv.DictReader(handle, delimiter='\t'))
    for entry in sources:
        actual = hashlib.sha256((root / 'data/raw' / entry['file']).read_bytes()).hexdigest()
        require(actual == entry['sha256'], f"Source hash mismatch: {entry['file']}")
    checks['source_hashes_verified'] = len(sources)
    parser = PDBParser(QUIET=True)
    models = list(parser.get_structure('nmr', root / 'data/raw/1L2Y.pdb'))
    sequences = [''.join(seq1(r.resname) for r in m.get_residues() if 'CA' in r) for m in models]
    fasta = ''.join(l.strip() for l in (root / 'data/prepared/1L2Y.fasta').read_text().splitlines() if not l.startswith('>'))
    require(all(s == fasta for s in sequences), 'NMR model / FASTA sequence mismatch')
    require(len(fasta) == 20 and len(models) == 38, 'Unexpected reference sequence or ensemble size')
    checks['nmr_models_with_matching_sequence'] = len(models)

    ref = Chem.RemoveHs(Chem.SDMolSupplier(str(root / 'data/raw/1iep_ligand.sdf'), removeHs=False)[0])
    ref_xyz = ref.GetConformer().GetPositions()
    raw = parser.get_structure('crystal', root / 'data/raw/1IEP.pdb')[0]['A']
    receptor = parser.get_structure('receptor', root / 'data/raw/1iep_receptorH.pdb')[0]['A']
    def heavy_atoms(chain):
        return {(r.id, r.resname, a.name): a.coord for r in chain if r.id[0] == ' ' for a in r if a.element != 'H'}
    raw_atoms, prepared_atoms = heavy_atoms(raw), heavy_atoms(receptor)
    require(set(raw_atoms) == set(prepared_atoms), 'Prepared receptor differs in heavy atom identity')
    require(all(np.linalg.norm(raw_atoms[k] - prepared_atoms[k]) < 1e-4 for k in raw_atoms), 'Prepared receptor heavy atoms moved')
    checks['receptor_heavy_atoms_verified_against_raw_pdb'] = len(raw_atoms)
    ligand = next(r for r in raw if r.resname == 'STI')
    crystal = np.array([a.coord for a in ligand if a.element != 'H'])
    require(len(crystal) == len(ref_xyz), 'Crystal / reference heavy atom count mismatch')
    distances = np.linalg.norm(ref_xyz[:, None] - crystal[None, :], axis=2)
    require(np.max(distances.min(axis=1)) < 1e-4, 'Reference SDF is not in the crystal coordinate frame')
    checks['ligand_heavy_atoms_verified_against_raw_pdb'] = len(crystal)

    with (root / 'results/docking/all_poses.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    table = {(int(r['exhaustiveness']), int(r['seed']), int(r['mode'])): r for r in rows}
    require(len(table) == len(rows), 'Duplicate docking table rows')
    errors, visited = [], set()
    for path in sorted((root / 'results/docking').glob('vina_e*_s*.sdf')):
        e, seed = map(int, re.fullmatch(r'vina_e(\d+)_s(\d+)', path.stem).groups())
        scores = [float(s) for s in re.findall(r'REMARK VINA RESULT:\s+(-?[\d.]+)', path.with_suffix('.pdbqt').read_text())]
        poses = list(Chem.SDMolSupplier(str(path), removeHs=False))
        require(len(poses) == len(scores), f'Pose/score mismatch: {path.name}')
        for mode, mol in enumerate(poses, 1):
            require(mol is not None, f'Unreadable pose {path.name}:{mode}')
            mol = Chem.RemoveHs(mol)
            require(mol.GetNumAtoms() == ref.GetNumAtoms() and mol.GetNumBonds() == ref.GetNumBonds(), 'Ligand graph size mismatch')
            maps = ref.GetSubstructMatches(mol, uniquify=False, useChirality=True, maxMatches=10000)
            require(0 < len(maps) < 10000, 'No exhaustive atom mapping available')
            xyz = mol.GetConformer().GetPositions()
            rmsd = min(float(np.sqrt(np.mean(np.sum((ref_xyz[list(m)] - xyz)**2, axis=1)))) for m in maps)
            key = (e, seed, mode)
            row = table[key]
            error = abs(rmsd - float(row['crystal_heavy_atom_rmsd_angstrom']))
            require(error < 1e-8, f'Independent RMSD mismatch: {key}')
            require(abs(scores[mode-1] - float(row['affinity_kcal_mol'])) < 1e-8, f'Score mismatch: {key}')
            errors.append(error)
            visited.add(key)
    require(visited == set(table), 'Missing or extra docking table rows')
    checks['docking_poses_independently_verified'] = len(visited)
    checks['maximum_rmsd_difference_angstrom'] = max(errors)
    checks['status'] = 'passed'
    checks['scope'] = 'Offline artifact checks; not a new docking, MD, or prediction run.'
    return checks


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, help='Optional JSON report; default is read-only')
    args = cli.parse_args()
    result = json.dumps(validate(), indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result)
