"""Record auditable inputs and observed inference behavior for a saved-MSA run."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('msa', type=Path)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    log = (args.run_dir / 'run.log').read_text()
    effective = re.search(r'Setting max_seq=(\d+), max_extra_seq=(\d+)', log)
    recycles = [int(x) for x in re.findall(r'recycle=(\d+)', log)]
    if effective is None or recycles != [0, 1, 2, 3] or 'Skipping ' in log:
        raise ValueError('Expected one actual inference with recycle 0–3, not cached output')
    config = json.loads((args.run_dir / 'config.json').read_text())
    summary = {'input_mode': 'saved A3M, no online MSA search', 'input_msa_sha256': sha256(args.msa), 'weights_model_1_ptm_sha256': sha256(args.weights), 'effective_max_seq': int(effective[1]), 'effective_max_extra_seq': int(effective[2]), 'recycles_observed': recycles, 'random_seed': config['random_seed'], 'model_type': config['model_type'], 'version': config['version'], 'outputs_sha256': {p.name: sha256(p) for p in sorted(args.run_dir.iterdir()) if p.name.endswith('.pdb') or '_scores_' in p.name}, 'limitation': 'The historical online-MSA route reported max_seq=4; loading its saved A3M reports max_seq=3. These routes are not assumed numerically equivalent.'}
    (args.run_dir / 'run_provenance.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
