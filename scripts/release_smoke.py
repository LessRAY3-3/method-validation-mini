#!/usr/bin/env python3
"""Read-only release checks using Python's standard library (no MD or inference).

Checks the required small artifacts shipped in Git. Large raw trajectories and
analysis logs are deliberately outside this check; use the release evidence pack.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def same(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)


def check_factorial(rows, contrasts, summaries):
    index = {(r['cell'], int(r['seed']), r['window_ns'], r['metric']): float(r['value']) for r in rows}
    require(len(index) == len(rows), 'Duplicate factorial endpoint')
    keys = {(r['window_ns'], r['metric']) for r in rows}
    require({r['window_ns'] for r in rows} == {'0-11','1-3','8-11'}, 'Wrong windows')
    require(len(index) == len(keys)*12, 'Incomplete factorial coverage')
    expected = {}
    expected_summaries = {}
    for window, metric in keys:
        for seed in (20261113,20261114,20261115):
            a,b,c,d = (index[(cell,seed,window,metric)] for cell in 'ABCD')
            for label,value in [('B-A',b-a),('D-C',d-c),('C-A',c-a),('D-B',d-b),('(D-C)-(B-A)',(d-c)-(b-a))]:
                expected[(seed,window,metric,label)] = value
        for cell in 'ABCD':
            values = [index[(cell,s,window,metric)] for s in (20261113,20261114,20261115)]
            expected_summaries[('cell',cell,window,metric)] = values
        for label in ('B-A','D-C','C-A','D-B','(D-C)-(B-A)'):
            expected_summaries[('contrast',label,window,metric)] = [expected[(s,window,metric,label)] for s in (20261113,20261114,20261115)]
    observed = {(int(r['seed']),r['window_ns'],r['metric'],r['contrast']): r['value'] for r in contrasts}
    require(len(observed)==len(contrasts) and observed.keys()==expected.keys(), 'Missing/duplicate contrast')
    require(all(same(observed[k],v) for k,v in expected.items()), 'Incorrect contrast arithmetic')
    actual = {(r['kind'],r['group'],r['window_ns'],r['metric']): r for r in summaries}
    require(len(actual)==len(summaries) and actual.keys()==expected_summaries.keys(), 'Missing/duplicate summary')
    for key,values in expected_summaries.items():
        r=actual[key]
        require(int(r['n_seed_branches'])==3, 'Wrong sample count')
        require(all(same(r[k],v) for k,v in [('mean',sum(values)/3),('minimum',min(values)),('maximum',max(values))]), 'Incorrect descriptive summary')
    return len(expected)


def validate(root=ROOT):
    counts = {}
    with (root/'data/raw/SOURCES.tsv').open() as f:
        sources=list(csv.DictReader(f,delimiter='\t'))
    for r in sources:
        require(sha(root/'data/raw'/r['file'])==r['sha256'], 'Source checksum: '+r['file'])
    counts['source_hashes']=len(sources)
    docking=root/'results/docking_benchmark_m3'; summary=json.loads((docking/'summary.json').read_text())
    runs=sorted(docking.glob('e*_s*/run.json'))
    require(len(runs)==15 and summary['complete_runs']==15, 'Need 15 formal docking runs')
    for p in runs:
        r=json.loads(p.read_text());require(r['status']=='complete','Docking incomplete')
        for name in ('poses.pdbqt','poses.sdf'):
            require(sha(p.parent/name)==r['output_sha256'][name], 'Docking pose checksum')
    poses=read_csv(docking/'all_poses.csv');require(len(poses)==summary['saved_poses']==57,'Need 57 poses')
    for group in summary['groups']:
        e=int(group['exhaustiveness']); top=[r for r in poses if int(r['exhaustiveness'])==e and int(r['rank'])==1]
        require(len(top)==5 and sum(float(r['crystal_rmsd_angstrom'])<=2 for r in top)==group['top1_successes'], 'Docking success count mismatch')
    counts['docking_pose_file_hashes']=30
    a=root/'results/md_factorial/analysis'; s=json.loads((a/'summary.json').read_text())
    require(s['status']=='complete' and s['n_cells']==4 and s['n_seed_branches_per_cell']==3,'Factorial incomplete')
    checked=omitted=0
    for rel,want in s['outputs_sha256'].items():
        if Path(rel).suffix in ('.gro','.log'):
            omitted+=1;continue
        require(sha(a/rel)==want,'Factorial small-artifact checksum: '+rel);checked+=1
    counts['factorial_small_output_hashes']=checked
    counts['archived_coordinate_and_log_files_outside_scope']=omitted
    counts['factorial_contrasts_recomputed']=check_factorial(read_csv(a/'per_trajectory_windows.csv'),read_csv(a/'seed_contrasts.csv'),read_csv(a/'descriptive_summary.csv'))
    p=json.loads((root/'output/pdf/report_provenance.json').read_text())
    require(sha(root/p['report'])==p['sha256'],'PDF checksum')
    for rel,want in p['input_sha256'].items():
        require(sha(root/rel)==want,'PDF source checksum: '+rel)
    require(sha(root/'scripts/build_report_pdf.py')==p['builder_sha256'],'PDF builder checksum')
    counts['pdf_input_hashes']=len(p['input_sha256'])
    audit=json.loads((root/'results/validation/md_factorial_m3_independent.json').read_text())
    require(audit['status']=='passed','M3 independent audit not passed')
    return {'status':'passed','checks':counts,'scope':'Required committed small artifacts, saved-pose hashes and summary arithmetic only; not new docking, dynamics, inference, raw-coordinate validation, or proof of convergence.'}


if __name__=='__main__':
    print(json.dumps(validate(),indent=2))
