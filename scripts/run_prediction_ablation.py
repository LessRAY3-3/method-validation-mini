"""Prepare a 3-MSA × 3-seed experiment; inference requires explicit --execute."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from prediction_provenance import sha256
from prediction_utils import fasta_sequence

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 2026, 20261110]
CONDITIONS = ['query_only', 'deduplicated_saved_msa', 'raw_saved_msa']


def parse_a3m(text, expected):
    records = []
    header = None
    for line in text.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        if line.startswith('>'):
            records.append([line, ''])
            header = line
        else:
            if header is None:
                raise ValueError('A3M sequence without header')
            records[-1][1] += line.strip()
    if not records or records[0][1] != expected:
        raise ValueError('A3M first query must match the expected sequence')
    for _, seq in records:
        aligned = ''.join(c for c in seq if not c.islower())
        if len(aligned) != len(expected) or any(c not in 'ACDEFGHIKLMNPQRSTVWYX-' for c in aligned):
            raise ValueError('Invalid A3M aligned columns')
    return records


def conditions_from_a3m(text, expected):
    records = parse_a3m(text, expected)
    unique, seen = [], set()
    for header, sequence in records:
        if sequence not in seen:
            unique.append((header, sequence))
            seen.add(sequence)
    def encode(items):
        return f'#{len(expected)}\t1\n' + ''.join(f'{h}\n{s}\n' for h, s in items)
    return {'query_only': encode(records[:1]), 'deduplicated_saved_msa': encode(unique), 'raw_saved_msa': text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/prediction_ablation')
    parser.add_argument('--weights-dir', type=Path, default=Path.home() / 'Library/Caches/colabfold')
    parser.add_argument('--execute', action='store_true', help='Explicitly run a prepared plan; default only prepares inputs and plan')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--colabfold-python', type=Path, default=ROOT / '.colabfold/bin/python')
    args = parser.parse_args()
    if args.output_dir.exists():
        if not (args.output_dir / 'plan.json').is_file():
            raise ValueError('Existing directory has no plan; choose a new directory')
        if args.execute:
            execute_plan(args)
        else:
            write_status(args.output_dir, 'prepared_or_completed; no inference requested')
            print('Existing experiment retained. No inference launched. Use --execute only when ready.')
        return
    source = ROOT / 'results/prediction/colabfold/1L2Y_model1.a3m'
    expected = fasta_sequence(ROOT / 'data/prepared/1L2Y.fasta')
    inputs = conditions_from_a3m(source.read_text(), expected)
    weight = args.weights_dir / 'params/params_model_1_ptm.npz'
    if not weight.is_file() or not (args.weights_dir / 'params/download_finished.txt').is_file():
        raise ValueError('Prepare existing weights first; this experiment never downloads weights')
    weight_hash = sha256(weight)
    inputs_dir = args.output_dir / 'inputs'
    inputs_dir.mkdir(parents=True)
    input_info = {}
    for condition, text in inputs.items():
        path = inputs_dir / f'{condition}.a3m'
        path.write_text(text)
        records = parse_a3m(text, expected)
        input_info[condition] = {'path': f'inputs/{path.name}', 'sha256': sha256(path), 'rows_including_query': len(records), 'unique_aligned_sequences': len(set(''.join(c for c in s if not c.islower()) for _, s in records))}
    plan = {'created_before_inference_utc': datetime.now(timezone.utc).isoformat(), 'sequence': expected, 'conditions_in_execution_order': CONDITIONS, 'seeds_in_execution_order': SEEDS, 'model_type': 'alphafold2_ptm', 'model_order': [1], 'num_models': 1, 'num_recycles': 3, 'num_seeds_per_run': 1, 'device': args.device.upper(), 'templates': False, 'relax': False, 'msa_search': 'none; every condition uses a local A3M via the same CLI input route', 'source_msa_sha256': sha256(source), 'weights_model_1_ptm_sha256': weight_hash, 'inputs': input_info, 'primary_metric': 'Median fitted CA RMSD over all 38 NMR models; paired condition differences within the same seed', 'secondary_metrics': ['all-NMR min/max and model1 RMSD', 'global-fit internal residues 3-18 and terminal residues 1,2,19,20 RMSD', 'per-residue CA displacement versus pLDDT', 'PAE and pTM'], 'interior_residues': list(range(3, 19)), 'terminal_residues': [1, 2, 19, 20], 'alignment': 'One global Kabsch fit over all 20 CA atoms for each NMR model, then subsets measured without refitting', 'analysis_limits': 'These are three paired random seeds on one known peptide, not independent targets or proof of generalization. NMR models are not statistical replicates. The interior subset is a positional definition, not a claimed hydrophobic core. No selecting the best seed or NMR model.'}
    (args.output_dir / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    write_status(args.output_dir, 'prepared; waiting for explicit --execute')
    if args.execute:
        execute_plan(args)
    else:
        print(f'Plan and inputs prepared: {args.output_dir}. No inference launched.')


def write_status(base, phase):
    entries = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            directory = base / 'runs' / condition / f'seed_{seed}'
            state = 'completed' if (directory / 'manifest.json').is_file() else ('incomplete' if directory.exists() else 'pending')
            entries.append({'condition': condition, 'seed': seed, 'state': state})
    (base / 'status.json').write_text(json.dumps({'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), 'runs': entries, 'no_background_worker_managed_by_this_file': True}, indent=2) + '\n')


def execute_plan(args):
    plan = json.loads((args.output_dir / 'plan.json').read_text())
    if plan['device'] != args.device.upper():
        raise ValueError('Requested device differs from the prepared plan; prepare a separate experiment')
    weight = args.weights_dir / 'params/params_model_1_ptm.npz'
    if not weight.is_file() or sha256(weight) != plan['weights_model_1_ptm_sha256']:
        raise ValueError('Weights do not match the prepared plan')
    if not (args.weights_dir / 'params/download_finished.txt').is_file():
        raise ValueError('Missing completed weight-download marker')
    plan_hash = sha256(args.output_dir / 'plan.json')
    env = os.environ.copy()
    env.update(JAX_PLATFORMS='cpu' if args.device == 'cpu' else 'cuda', JAX_COMPILATION_CACHE_DIR=str(ROOT / '.cache/colabfold_jax'), MPLCONFIGDIR=str(ROOT / '.cache/matplotlib'), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    for condition in CONDITIONS:
        for seed in SEEDS:
            if (args.output_dir / 'STOP').exists():
                write_status(args.output_dir, 'paused between runs by STOP marker')
                print('STOP marker found; no further inference launched.')
                return
            run_dir = args.output_dir / 'runs' / condition / f'seed_{seed}'
            if (run_dir / 'manifest.json').is_file():
                from analyze_prediction_ablation import verify_run
                verify_run(args.output_dir, plan, condition, seed)
                continue
            if run_dir.exists():
                raise ValueError(f'Incomplete run directory exists: {run_dir}; preserve it and prepare a new experiment')
            run_dir.mkdir(parents=True)
            msa = args.output_dir / plan['inputs'][condition]['path']
            if sha256(msa) != plan['inputs'][condition]['sha256']:
                raise ValueError('Prepared MSA changed')
            command = [str(args.colabfold_python), str(ROOT / '.colabfold/bin/colabfold_batch'), str(msa), str(run_dir), '--data', str(args.weights_dir), '--model-type', 'alphafold2_ptm', '--model-order', '1', '--num-models', '1', '--num-seeds', '1', '--random-seed', str(seed), '--num-recycle', '3', '--rank', 'plddt', '--compile-mode', 'fast']
            # Use the selected environment's module rather than an architecture-specific entrypoint.
            command[1:2] = ['-m', 'colabfold.batch']
            started = datetime.now(timezone.utc).isoformat()
            print(f'Running {condition} seed={seed}', flush=True)
            write_status(args.output_dir, f'running {condition} seed={seed}; sequential')
            with (run_dir / 'run.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            manifest = {'condition': condition, 'seed': seed, 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'command': command, 'plan_sha256': plan_hash, 'input_sha256': sha256(msa), 'weights_sha256': plan['weights_model_1_ptm_sha256'], 'output_sha256': {p.name: sha256(p) for p in sorted(run_dir.iterdir()) if p.is_file()}}
            (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_prediction_ablation.py'), '--experiment-dir', str(args.output_dir)], env=env, check=True)
    write_status(args.output_dir, 'completed; no running inference')


if __name__ == '__main__':
    main()
