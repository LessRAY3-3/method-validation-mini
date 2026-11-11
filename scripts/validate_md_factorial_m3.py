#!/usr/bin/env python3
"""Independent cached-coordinate audit; no MD, GROMACS, or original output writes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import Bio
from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np

BASE = Path(__file__).resolve().parents[1] / 'setup/m3_factorial_validation'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(task):
    cell, seed = task
    start = time.time()
    path = BASE / cell / str(seed)
    lines = (path / 'protein_whole.gro').read_text().splitlines()
    frames, times, ids = [], [], None
    k = 0
    while k < len(lines):
        times.append(float(lines[k].split('t=')[-1].split()[0]))
        n = int(lines[k + 1]); records = lines[k+2:k+2+n]
        current = [(int(r[:5]), r[5:10].strip(), r[10:15].strip()) for r in records]
        if ids is None: ids = current
        assert current == ids
        frames.append([[float(r[20:28])*10, float(r[28:36])*10, float(r[36:44])*10] for r in records])
        k += n + 3
    xyz = np.asarray(frames); times = np.asarray(times)
    assert np.array_equal(times, np.arange(0, 11001, 10))
    assert len(ids) == 304
    ca = xyz[:, [i for i, a in enumerate(ids) if a[2] == 'CA']]
    refs = []; current = []
    for line in (BASE / '1L2Y.pdb').read_text().splitlines():
        if line.startswith('MODEL'): current = []
        elif line.startswith('ATOM') and line[12:16].strip() == 'CA':
            current.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith('ENDMDL'): refs.append(np.asarray(current))
    assert len(refs) == 38 and all(x.shape == (20, 3) for x in refs)
    with (path / 'timeseries.csv').open() as f: series = list(csv.DictReader(f))
    metric_errors = {}
    ii, jj = np.triu_indices(20, 3)
    reference_d = np.linalg.norm(refs[0][ii] - refs[0][jj], axis=1)
    actual_d = np.linalg.norm(ca[:, ii] - ca[:, jj], axis=2)
    for cutoff in (7,8,9):
        mask = reference_d <= cutoff
        value = np.mean(actual_d[:, mask] <= 1.2 * reference_d[mask], axis=1)
        key = f'q_ca_{cutoff}A'
        metric_errors[key] = float(np.max(np.abs(value - [float(r[key]) for r in series])))
    for a,b in ((6,12),(6,18),(3,19)):
        for mode in ('heavy', 'sidechain'):
            groups = [[i for i,(res,_,atom) in enumerate(ids) if res == rr and not atom.lstrip('0123456789').startswith('H') and
                       (mode == 'heavy' or atom not in {'N','CA','C','O','OXT','OT1','OT2'})] for rr in (a,b)]
            value = np.linalg.norm(xyz[:, groups[0], None] - xyz[:, None, groups[1]], axis=-1).min(axis=(1,2))
            key = f'{mode}_min_{a}_{b}_angstrom'
            metric_errors[key] = float(np.max(np.abs(value - [float(r[key]) for r in series])))
    with (BASE / 'all_reference_window_results.csv').open() as f:
        expected = {(int(r['nmr_model']),r['window_ns']):r for r in csv.DictReader(f) if int(r['seed']) == seed and r['cell'] == cell}
    core_pairs = (ii >= 2) & (jj < 18)
    errors = []; count = 0
    for model, ref in enumerate(refs,1):
        values = []
        for selection in (slice(None),slice(2,18)):
            rmsd = []
            for frame in ca:
                fit = SVDSuperimposer(); fit.set(ref[selection],frame[selection]); fit.run()
                rmsd.append(fit.get_rms())
            values.append(np.array(rmsd))
        maes = np.mean(np.abs(actual_d - np.linalg.norm(ref[ii] - ref[jj],axis=1)),axis=1)
        core_maes = np.mean(np.abs(actual_d[:,core_pairs] - np.linalg.norm(ref[ii[core_pairs]] - ref[jj[core_pairs]],axis=1)),axis=1)
        for low, high in ((0,11),(1,3),(8,11)):
            mask = (times >= low*1000) & ((times <= high*1000) if high == 11 else (times < high*1000))
            row = expected[(model,f'{low}-{high}')]
            for key,val in zip(('mean_ca_rmsd_angstrom','mean_interior_ca_rmsd_angstrom','mean_nonlocal_distance_mae_angstrom','mean_core_nonlocal_distance_mae_angstrom'),(*values,maes,core_maes)):
                errors.append(abs(float(np.mean(val[mask])) - float(row[key]))); count += 1
        if model == 1:
            metric_errors['ca_rmsd_nmr1_angstrom'] = float(np.max(np.abs(values[0] - [float(r['ca_rmsd_nmr1_angstrom']) for r in series])))
            metric_errors['core_ca_rmsd_nmr1_angstrom'] = float(np.max(np.abs(values[1] - [float(r['core_ca_rmsd_nmr1_angstrom']) for r in series])))
            metric_errors['core_nonlocal_distance_mae_nmr1_angstrom'] = float(np.max(np.abs(core_maes - [float(r['core_nonlocal_distance_mae_nmr1_angstrom']) for r in series])))
    a = [i for i,(res,_,atom) in enumerate(ids) if res == 3 and not atom.lstrip('0123456789').startswith('H') and atom not in {'N','CA','C','O','OXT','OT1','OT2'}]
    b = [i for i,(res,_,atom) in enumerate(ids) if res == 19 and not atom.lstrip('0123456789').startswith('H') and atom not in {'N','CA','C','O','OXT','OT1','OT2'}]
    side = np.linalg.norm(xyz[:,a,None] - xyz[:,None,b],axis=-1).min(axis=(1,2))
    metric_errors['y3p19_fraction_above_6A'] = float(np.max(np.abs((side > 6).astype(float) - [float(r['y3p19_fraction_above_6A']) for r in series])))
    extra_errors = []; extra_counts = {}
    for filename in ('all_q_windows.csv','selected_contact_occupancies.csv','threshold_sensitivity.csv'):
        with (BASE/filename).open() as f: rows = [r for r in csv.DictReader(f) if r['cell']==cell and int(r['seed'])==seed]
        extra_counts[filename] = len(rows)
        for row in rows:
            low,high = map(float,row['window_ns'].split('-'))
            mask = (times >= low*1000) & ((times <= high*1000) if high == 11 else (times < high*1000))
            if filename == 'all_q_windows.csv':
                selected = reference_d <= float(row['native_cutoff_A'])
                assert int(selected.sum()) == int(row['native_pair_count'])
                actual = np.mean(actual_d[mask][:,selected] <= float(row['retention_factor'])*reference_d[selected])
                extra_errors.append(abs(actual-float(row['mean_q'])))
            elif filename == 'selected_contact_occupancies.csv':
                ia,ib = int(row['residue_i'])-1,int(row['residue_j'])-1
                refdistance = np.linalg.norm(refs[0][ia]-refs[0][ib])
                if refdistance > float(row['native_cutoff_A']):
                    assert row['selection_status']=='excluded' and row['occupancy'] in ('','null')
                else:
                    assert row['selection_status']=='included'
                    actual = np.mean(np.linalg.norm(ca[mask,ia]-ca[mask,ib],axis=-1) <= float(row['retention_factor'])*refdistance)
                    extra_errors.append(abs(actual-float(row['occupancy'])))
            else:
                above = side[mask] > float(row['threshold_A']); tt=times[mask]
                extra_errors.append(abs(float(np.mean(above))-float(row['sample_fraction_above'])))
                assert int(above.sum()) == int(row['n_samples_above'])
                longest=0; beginning=None
                for t,flag in zip(tt,above):
                    if flag and beginning is None: beginning=t
                    if flag: longest=max(longest,t-beginning)
                    else: beginning=None
                assert longest == int(row['longest_sampled_span_ps'])
    assert max(extra_errors) < 1e-9
    maximum = max(max(metric_errors.values()), max(errors))
    assert maximum < 1e-9, (seed, maximum)
    return {'cell':cell,'seed':seed,'extra_row_checks':extra_counts,'max_extra_error':max(extra_errors), 'frames':len(times), 'protein_atoms':len(ids), 'reference_scalar_checks':count,
            'max_reference_error_A':max(errors), 'timeseries_max_errors':metric_errors,
            'elapsed_seconds':time.time()-start, 'status':'passed'}


def main():
    out = BASE / 'validation_result.json'
    if out.exists(): raise SystemExit('Refusing to overwrite audit')
    inputs = {str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.suffix in ('.gro','.csv','.pdb')}
    metadata = {'status':'running','started_unix':time.time(),'pid':os.getpid(),
                'platform':platform.platform(),'machine':platform.machine(), 'python':sys.version,
                'numpy':np.__version__,'biopython':Bio.__version__, 'workers':6,
                'script_sha256':sha(Path(__file__)), 'input_sha256':inputs,
                'scope':'Independent parser and Bio.SVDSuperimposer numerical audit of cached whole-protein GRO; not dynamics or PBC re-export verification'}
    out.write_text(json.dumps(metadata,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(worker,[(c,s) for c in 'ABCD' for s in range(20261113,20261116)]))
    assert inputs == {str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.suffix in ('.gro','.csv','.pdb')}
    metadata.update(status='passed',results=results,completed_unix=time.time())
    out.write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'status':'passed','reference_scalar_checks':sum(r['reference_scalar_checks'] for r in results),
                      'max_reference_error_A':max(r['max_reference_error_A'] for r in results)}))


if __name__ == '__main__': main()
