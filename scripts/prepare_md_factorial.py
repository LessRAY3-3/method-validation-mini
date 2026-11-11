#!/usr/bin/env python3
"""Prepare the preselected NMR model 24 in an isolated directory, no production MD."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atoms(path):
    lines = path.read_text().splitlines()
    return [(int(s[:5]), s[5:10].strip(), s[10:15].strip(),
             [float(s[20:28]), float(s[28:36]), float(s[36:44])])
            for s in lines[2:2 + int(lines[1])]]


def protein_topology(path):
    text = path.read_text()
    # Compare all parameters in the protein molecule, not just atom labels.
    text = text[text.index('[ moleculetype ]'):text.index('; Include Position restraint file')]
    return [line.split(';')[0].split() for line in text.splitlines()
            if line.split(';')[0].strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gmx', required=True)
    args = ap.parse_args()
    out = ROOT / 'setup/md_factorial/model24'
    if out.exists():
        raise SystemExit('Refusing to overwrite existing preparation directory')
    out.mkdir(parents=True)
    gmx = str(Path(args.gmx).resolve())
    source = ROOT / 'data/raw/1L2Y.pdb'
    provenance = {'status': 'preparing', 'model': 24, 'ion_seed': 20261111,
                  'selection': 'maximum model1-fitted CA3-18 RMSD among38 NMR models; selected before new MD',
                  'selection_rmsd_A': 0.5396169716066097,
                  'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                      (source, ROOT / 'config/ions.mdp', ROOT / 'config/minim.mdp',
                       ROOT / 'results/md/topol.top', ROOT / 'results/md/minim.gro')},
                  'script_sha256': sha(__file__), 'gmx_binary_sha256': sha(gmx)}
    selected = []
    active = False
    for line in source.read_text().splitlines():
        if line.startswith('MODEL'):
            active = int(line[10:14]) == 24
        elif line.startswith('ENDMDL') and active:
            break
        elif active and line.startswith(('ATOM', 'TER')):
            selected.append(line)
    if not selected:
        raise RuntimeError('Model24 missing')
    (out / 'model24.pdb').write_text('\n'.join(selected + ['END']) + '\n')
    def run(argv, stdin=None):
        with (out / 'commands.jsonl').open('a') as f:
            f.write(json.dumps({'argv': argv, 'stdin': stdin, 'time_unix': time.time()}) + '\n')
        result = subprocess.run(argv, cwd=out, input=stdin, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with (out / 'preparation.log').open('a') as f:
            f.write(result.stdout)
        if result.returncode:
            raise RuntimeError(f'Command failed: {argv}; inspect preparation.log')
        return result.stdout
    provenance['gromacs_version'] = run([gmx, '--version'])
    (out / 'preparation_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    run([gmx, 'pdb2gmx', '-f', 'model24.pdb', '-o', 'prepared.gro', '-p', 'topol.top',
         '-i', 'posre.itp', '-ff', 'amber99sb-ildn', '-water', 'tip3p', '-ignh'])
    ref = atoms(ROOT / 'results/md/minim.gro')[:304]
    prot = atoms(out / 'prepared.gro')
    if len(prot) != 304 or [a[:3] for a in prot] != [a[:3] for a in ref]:
        raise RuntimeError('Protein atom/residue mapping differs from original304 atoms')
    if protein_topology(out / 'topol.top') != protein_topology(ROOT / 'results/md/topol.top'):
        raise RuntimeError('Protein topology parameters differ from original')
    run([gmx, 'editconf', '-f', 'prepared.gro', '-o', 'boxed.gro', '-c', '-d', '1.0', '-bt', 'cubic'])
    run([gmx, 'solvate', '-cp', 'boxed.gro', '-cs', 'spc216', '-o', 'solvated.gro', '-p', 'topol.top'])
    ionlog = run([gmx, 'grompp', '-f', str(ROOT / 'config/ions.mdp'), '-c', 'solvated.gro',
                 '-p', 'topol.top', '-o', 'ions.tpr', '-po', 'ions_processed.mdp', '-maxwarn', '1'])
    # The only accepted warning is the expected charge before neutralization.
    warnings = re.findall(r'^WARNING \d+.*?(?=\n\n)', ionlog, re.S | re.M)
    if any('net charge' not in w.lower() for w in warnings):
        raise RuntimeError('Unexpected ion-grompp warning; inspect log')
    run([gmx, 'genion', '-s', 'ions.tpr', '-o', 'ionized.gro', '-p', 'topol.top',
         '-pname', 'NA', '-nname', 'CL', '-neutral', '-seed', '20261111'], 'SOL\n')
    run([gmx, 'grompp', '-f', str(ROOT / 'config/minim.mdp'), '-c', 'ionized.gro',
         '-p', 'topol.top', '-o', 'minim.tpr', '-po', 'minim_processed.mdp'])
    run([gmx, 'mdrun', '-s', 'minim.tpr', '-deffnm', 'minim', '-ntmpi', '1', '-ntomp', '4',
         '-nb', 'cpu', '-pme', 'cpu'])
    log = (out / 'minim.log').read_text()
    if not re.search(r'converged to Fmax <', log) or re.search(r'LINCS WARNING|Fatal error', log, re.I):
        raise RuntimeError('Minimization did not satisfy requested convergence')
    minimized = atoms(out / 'minim.gro')
    if [a[:3] for a in minimized[:304]] != [a[:3] for a in ref]:
        raise RuntimeError('Minimized protein mapping changed')
    def ca_rmsd(a, b):
        aa = np.array([r[3] for r in a if r[2] == 'CA' and 3 <= r[0] <= 18]) * 10
        bb = np.array([r[3] for r in b if r[2] == 'CA' and 3 <= r[0] <= 18]) * 10
        aa -= aa.mean(0); bb -= bb.mean(0)
        u, s, vt = np.linalg.svd(aa.T @ bb)
        rot = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        return float(np.sqrt(np.mean(np.sum((aa @ rot - bb)**2, axis=1))))
    provenance.update(status='complete', protein_atoms=304, protein_topology_identical=True,
                      total_atoms=len(minimized), water_molecules=sum(a[1] == 'SOL' and a[2] == 'OW' for a in minimized),
                      ions=[a[:3] for a in minimized if a[1] in ('CL', 'NA')],
                      box_nm=(out / 'minim.gro').read_text().splitlines()[-1].split(),
                      minimized_CA3_18_rmsd_to_original_minim_A=ca_rmsd(minimized[:304], ref),
                      own_prepared_to_minimized_CA3_18_rmsd_A=ca_rmsd(minimized[:304], prot),
                      completed_unix=time.time(), output_sha256={n: sha(out / n) for n in
                          ('model24.pdb', 'prepared.gro', 'minim.gro', 'minim.tpr', 'topol.top', 'posre.itp',
                           'ions_processed.mdp', 'minim_processed.mdp', 'minim.log', 'commands.jsonl')})
    (out / 'preparation_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
