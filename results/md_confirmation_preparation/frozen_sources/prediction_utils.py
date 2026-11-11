"""Small, explicit coordinate/sequence utilities shared by prediction analyses."""
from pathlib import Path
import numpy as np

AA = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(), 'ARNDCQEGHILKMFPSTWYV'))


def fasta_sequence(path):
    lines = Path(path).read_text().splitlines()
    if sum(line.startswith('>') for line in lines) != 1:
        raise ValueError('Expected exactly one FASTA sequence')
    return ''.join(line.strip() for line in lines if not line.startswith('>'))


def ca_models(path, expected_sequence):
    """Read every PDB model, reject missing/extra/reordered residues or altloc ambiguity."""
    models, atoms, seen = [], [], set()
    model_number = 1
    def finish():
        if not atoms:
            return
        sequence = ''.join(atom[1] for atom in atoms)
        if sequence != expected_sequence:
            raise ValueError(f'{path}, model {model_number}: sequence mismatch: {sequence!r} != {expected_sequence!r}')
        models.append((model_number, np.array([a[2] for a in atoms]), np.array([a[3] for a in atoms])))
    for line in Path(path).read_text().splitlines():
        if line.startswith('MODEL'):
            model_number = int(line[10:14])
        elif line.startswith('ENDMDL'):
            finish()
            atoms, seen = [], set()
        elif line.startswith('ATOM') and line[12:16].strip() == 'CA':
            if line[16] not in (' ', 'A'):
                continue
            key = (line[21], line[22:26], line[26])
            if key in seen:
                raise ValueError(f'{path}: duplicate CA {key}')
            seen.add(key)
            atoms.append((key, AA.get(line[17:20], 'X'), [float(line[i:i+8]) for i in (30, 38, 46)], float(line[60:66])))
    finish()
    if not models:
        raise ValueError(f'{path}: no CA coordinates')
    return models


def fitted_ca_rmsd(mobile, reference):
    """Kabsch least-squares CA RMSD in Å; reject reflections and unequal arrays."""
    mobile, reference = np.asarray(mobile), np.asarray(reference)
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3 or len(mobile) < 3:
        raise ValueError('Expected matching N×3 coordinate arrays with N >= 3')
    if not np.isfinite(mobile).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite coordinates')
    x, y = mobile - mobile.mean(0), reference - reference.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return float(np.sqrt(np.mean(np.sum((x @ u @ correction @ vt - y) ** 2, axis=1))))
