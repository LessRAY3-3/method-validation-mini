"""Compare each prediction with every experimental NMR model (CA Kabsch RMSD)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prediction_utils import ca_models, fasta_sequence, fitted_ca_rmsd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--colabfold-dir', type=Path, default=ROOT / 'results/prediction/colabfold')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/prediction/nmr_comparison')
    args = parser.parse_args()
    sequence = fasta_sequence(ROOT / 'data/prepared/1L2Y.fasta')
    reference = ROOT / 'data/raw/1L2Y.pdb'
    models = ca_models(reference, sequence)
    predictions = [('ESMFold', ROOT / 'results/prediction/esmfold_1L2Y.pdb')]
    predictions += [(p.stem, p) for p in sorted(args.colabfold_dir.glob('*unrelaxed*.pdb'))]
    if len(predictions) < 2:
        raise ValueError('No ColabFold PDB found')
    rows, summaries = [], []
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, path in predictions:
        predicted = ca_models(path, sequence)
        if len(predicted) != 1:
            raise ValueError(f'Expected exactly one predicted model: {path}')
        rmsds = np.array([fitted_ca_rmsd(predicted[0][1], coords) for _, coords, _ in models])
        for (number, _, _), value in zip(models, rmsds):
            rows.append({'prediction': name, 'nmr_model': number, 'ca_rmsd_angstrom': float(value)})
        summaries.append({'prediction': name, 'prediction_source': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), 'reference_models': len(models), 'residues': len(sequence), 'model1_rmsd_angstrom': float(rmsds[[m[0] for m in models].index(1)]), 'minimum_rmsd_angstrom': float(rmsds.min()), 'closest_nmr_model': models[int(rmsds.argmin())][0], 'median_rmsd_angstrom': float(np.median(rmsds)), 'maximum_rmsd_angstrom': float(rmsds.max()), 'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        label = 'ColabFold ' + name.split('_rank_')[-1] if name != 'ESMFold' else name
        ax.plot([m[0] for m in models], rmsds, marker='o', markersize=3, label=label)
    ax.set(xlabel='Experimental NMR model number', ylabel='Fitted Cα RMSD (Å)', xticks=[1, 5, 10, 15, 20, 25, 30, 35, len(models)])
    ax.legend(fontsize=7)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / 'prediction_vs_nmr.png', dpi=200)
    plt.close(fig)
    with (args.output_dir / 'per_model.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {'method': 'One-to-one sequence-validated CA mapping; Kabsch rotation and translation; no reflection; angstrom', 'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(), 'reference_model_count': len(models), 'predictions': summaries, 'limitations': 'NMR models are an experimental ensemble, not independent replicates. Report the full distribution, not only the nearest model. This known historical structure is not a held-out generalization benchmark, nor evidence of binding mechanism. Fitted protein RMSD deliberately removes rotation/translation, unlike redocking RMSD.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
