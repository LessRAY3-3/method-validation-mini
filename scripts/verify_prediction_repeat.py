"""Verify current files against provenance and compare two actual saved-MSA runs."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from prediction_provenance import sha256
ROOT = Path(__file__).resolve().parents[1]


def read_run(directory):
    paths = list(directory.glob('*scores*.json'))
    if len(paths) != 1:
        raise ValueError('Expected one scores file per run')
    return json.loads(paths[0].read_text())


def verify_evidence(directory):
    """Rehash actual outputs; stale provenance must never certify changed files."""
    provenance = json.loads((directory / 'run_provenance.json').read_text())
    required = {'input_msa_sha256', 'weights_model_1_ptm_sha256', 'random_seed', 'model_type', 'effective_max_seq', 'effective_max_extra_seq', 'version', 'recycles_observed', 'outputs_sha256'}
    if required - provenance.keys():
        raise ValueError(f'{directory}: incomplete provenance')
    expected = provenance['outputs_sha256']
    actual_files = {p.name: p for p in directory.iterdir() if p.name.endswith('.pdb') or '_scores_' in p.name}
    if set(expected) != set(actual_files) or len([p for p in actual_files if p.endswith('.pdb')]) != 1 or len([p for p in actual_files if '_scores_' in p]) != 1:
        raise ValueError(f'{directory}: output inventory mismatch')
    actual_hashes = {name: sha256(path) for name, path in actual_files.items()}
    if actual_hashes != expected:
        raise ValueError(f'{directory}: output SHA-256 mismatch; files changed after provenance was recorded')
    log = (directory / 'run.log').read_text()
    observed_recycles = [int(x) for x in re.findall(r'recycle=(\d+)', log)]
    effective = re.search(r'Setting max_seq=(\d+), max_extra_seq=(\d+)', log)
    logged_version = re.search(r'Running colabfold (\S+)', log)
    if observed_recycles != [0, 1, 2, 3] or provenance['recycles_observed'] != observed_recycles or 'Skipping ' in log:
        raise ValueError(f'{directory}: missing actual inference or recycle evidence')
    if effective is None or [int(effective[1]), int(effective[2])] != [provenance['effective_max_seq'], provenance['effective_max_extra_seq']]:
        raise ValueError(f'{directory}: effective MSA parameters disagree with log')
    config = json.loads((directory / 'config.json').read_text())
    if logged_version is None or logged_version[1] != provenance['version']:
        raise ValueError(f'{directory}: version disagrees with log')
    for key in ('version', 'random_seed', 'model_type'):
        if config[key] != provenance[key]:
            raise ValueError(f'{directory}: {key} disagrees with config')
    if config['num_recycles'] != 3 or config['num_models'] != 1 or config['num_seeds'] != 1:
        raise ValueError(f'{directory}: incompatible inference configuration')
    return provenance, actual_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/prediction/reproducibility_summary.json')
    args = parser.parse_args()
    evidence = [verify_evidence(p) for p in (args.first, args.second)]
    provenance = [item[0] for item in evidence]
    first, second = read_run(args.first), read_run(args.second)
    baseline = read_run(ROOT / 'results/prediction/colabfold')
    equality = {key: bool(np.array_equal(first[key], second[key])) for key in ('plddt', 'pae', 'ptm')}
    compared = ('input_msa_sha256', 'weights_model_1_ptm_sha256', 'random_seed', 'model_type', 'effective_max_seq', 'effective_max_extra_seq', 'version', 'recycles_observed')
    inputs_equal = all(provenance[0][key] == provenance[1][key] for key in compared)
    equality['pdb_sha256'] = sorted(v for k, v in evidence[0][1].items() if k.endswith('.pdb')) == sorted(v for k, v in evidence[1][1].items() if k.endswith('.pdb'))
    def stats(data):
        return {'mean_plddt': float(np.mean(data['plddt'])), 'mean_pae_angstrom': float(np.mean(data['pae'])), 'maximum_pae_angstrom': float(np.max(data['pae'])), 'ptm_rounded_in_json': data['ptm']}
    passed = inputs_equal and all(equality.values())
    historical_equal = all(np.array_equal(first[key], baseline[key]) for key in ('plddt', 'pae', 'ptm'))
    result = {'first_directory': str(args.first), 'second_directory': str(args.second), 'current_output_hashes_match_provenance': True, 'actual_recycles_and_version_match_logs': True, 'compared_input_fields': list(compared), 'fixed_inputs_equal': inputs_equal, 'repeat_equal': equality, 'passed': passed, 'historical_online': stats(baseline), 'fixed_msa_offline': stats(first), 'observed_max_seq': [p['effective_max_seq'] for p in provenance], 'historical_and_offline_scores_identical': historical_equal, 'interpretation': ('The two supplied saved-MSA runs agree on the compared inputs, scores and current PDB bytes.' if passed else 'The supplied runs do not satisfy the repeatability checks.') + (' Their scores match the historical archived run.' if historical_equal else ' Their scores differ from the historical archived run; do not describe this as numerical reproduction of that route.') + ' This check validates the retained output/log/config evidence, not independent retrieval of unavailable original input files. pTM in scores JSON is rounded to two decimals; logs retain more precision.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
