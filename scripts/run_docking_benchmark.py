"""Execute a predeclared 3 x 5 paired-seed Vina experiment in a new directory.

Usage: .conda/bin/python scripts/run_docking_benchmark.py --run [--workers 1|2]
Without --run, only print status; no calculations are started.
Existing successful runs are reused only if protocol and all output hashes match.
No historical docking runs are reused. --analyze-only never invokes Vina.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/docking_benchmark'
SEEDS = [42, 2026, 20261110, 20261111, 20261112]
EFFORTS = [1, 8, 32]
INPUTS = ['data/prepared/1iep_receptor.pdbqt', 'data/prepared/1iep_ligand.pdbqt',
          'config/1iep_box.txt', 'data/raw/1iep_ligand.sdf', 'data/raw/1iep_receptorH.pdb']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def protocol():
    return {
        'study': '1IEP paired-seed search effort experiment; distinct from historical four runs',
        'exhaustiveness': EFFORTS, 'paired_seeds': SEEDS, 'cpu_per_run': 1,
        'num_modes': 9, 'energy_range_kcal_mol': 3,
        'success_definition': 'symmetry-corrected heavy atom crystal RMSD <= 2.0 angstrom; no alignment or centering',
        'primary_endpoint': 'top-scored saved pose success fraction, n=5 per exhaustiveness',
        'secondary_endpoint': 'at least one saved pose success fraction; not all searched conformations',
        'contact_definition': 'residue contact if any receptor/ligand heavy atom distance <= 4.0 angstrom; compare residue sets to crystal',
        'analysis': 'Report all 15 runs, all saved poses, paired outcomes, Wilson 95% intervals (descriptive, n=5); no seed or threshold selection after results',
        'timing_limit': 'Wall-clock timing includes contention from concurrent local work; not a hardware performance benchmark',
        'generalization_limit': 'One prepared co-crystal system; seed success fractions do not establish cross-target accuracy',
        'inputs_sha256': {f: sha(ROOT / f) for f in INPUTS},
        'vina_version': subprocess.check_output([str(ROOT / '.conda/bin/vina'), '--version'], text=True).strip(),
        'vina_binary_sha256': sha(ROOT / '.conda/bin/vina'),
    }


def run_one(e, seed):
    folder = OUT / f'e{e}_s{seed}'
    folder.mkdir(exist_ok=True)
    record = folder / 'run.json'
    if record.exists():
        old = json.loads(record.read_text())
        if old.get('status') == 'complete' and all(sha(folder / name) == value for name, value in old['output_sha256'].items()):
            return old
        if old.get('status') != 'pending':
            raise RuntimeError(f'Incomplete/changed run preserved for inspection: {folder}; inspect it before resuming')
    rel = folder
    cmd = ['.conda/bin/vina', '--receptor', INPUTS[0], '--ligand', INPUTS[1], '--config', INPUTS[2],
           '--scoring', 'vina', '--exhaustiveness', str(e), '--num_modes', '9', '--energy_range', '3',
           '--seed', str(seed), '--cpu', '1', '--out', str(rel / 'poses.pdbqt')]
    data = {'exhaustiveness': e, 'seed': seed, 'command': cmd, 'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
    write_json(record, data)
    start = time.perf_counter()
    with (folder / 'vina.log').open('w') as log:
        p = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    data.update(vina_exit_code=p.returncode, vina_elapsed_seconds=time.perf_counter()-start)
    if p.returncode == 0:
        with (folder / 'export.log').open('w') as log:
            p = subprocess.run([str(ROOT / '.conda/bin/python'), str(ROOT / '.conda/bin/mk_export.py'), str(folder/'poses.pdbqt'), '-s', str(folder/'poses.sdf')], stdout=log, stderr=subprocess.STDOUT)
        data['export_exit_code'] = p.returncode
    data['status'] = 'complete' if p.returncode == 0 else 'failed'
    data['finished_utc'] = datetime.now(timezone.utc).isoformat()
    data['output_sha256'] = {name:sha(folder/name) for name in ['poses.pdbqt','poses.sdf','vina.log','export.log'] if (folder/name).exists()}
    write_json(record, data)
    print(f'e={e} seed={seed}: {data["status"]}, {data["vina_elapsed_seconds"]:.1f}s', flush=True)
    return data


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='Study directory; relative paths resolve from the repository root')
    parser.add_argument('--workers', type=int, choices=[1,2], default=1)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Explicitly start/resume pending calculations')
    mode.add_argument('--analyze-only', action='store_true')
    args=parser.parse_args()
    if args.output_dir is not None:
        OUT = args.output_dir.resolve() if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    if not args.run and not args.analyze_only:
        records=[]
        for seed in SEEDS:
            for e in EFFORTS:
                path=OUT/f'e{e}_s{seed}'/'run.json'
                records.append(json.loads(path.read_text()) if path.exists() else {'exhaustiveness':e,'seed':seed,'status':'pending'})
        print(json.dumps({'planned_runs':15,'counts':{status:sum(r['status']==status for r in records) for status in sorted({r['status'] for r in records})},'instruction':'Use --run to explicitly resume. Standard Vina uses CPU, not CUDA.'},indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    expected=protocol()
    path=OUT/'protocol.json'
    if path.exists():
        stored=json.loads(path.read_text())
        assert stored['protocol'] == expected, 'Protocol/input changed; refusing to mix experiments'
    else:
        if args.analyze_only:
            raise FileNotFoundError(path)
        write_json(path, {'declared_before_runs_utc':datetime.now(timezone.utc).isoformat(), 'protocol':expected})
    if not args.analyze_only:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_one,e,seed) for seed in SEEDS for e in EFFORTS]
            records=[f.result() for f in futures]
        if any(r['status']!='complete' for r in records):
            raise RuntimeError('Execution failures retained; inspect run.json and logs before analysis')
    subprocess.run([sys.executable, str(ROOT/'scripts/analyze_docking_benchmark.py'), '--output-dir', str(OUT)], check=True)


if __name__=='__main__':
    main()
