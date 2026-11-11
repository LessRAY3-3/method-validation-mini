"""Analyze archived or fresh predictions without overwriting prediction inputs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prediction_utils import ca_models, fasta_sequence

ROOT = Path(__file__).resolve().parents[1]


def validate_scores(data, length):
    plddt, pae = np.asarray(data['plddt'], float), np.asarray(data['pae'], float)
    if plddt.shape != (length,) or pae.shape != (length, length):
        raise ValueError('pLDDT/PAE shape does not match the sequence length')
    if not np.isfinite(plddt).all() or not np.isfinite(pae).all() or np.any((plddt < 0) | (plddt > 100)) or np.any(pae < 0):
        raise ValueError('Invalid pLDDT or PAE values')
    return plddt, pae


def plot_plddt(scores, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    residues = np.arange(1, len(scores) + 1)
    ax.plot(residues, scores, marker='o')
    for low, high, color in [(90, 100, 'navy'), (70, 90, 'skyblue'), (50, 70, 'gold'), (0, 50, 'orange')]:
        ax.axhspan(low, high, color=color, alpha=.1)
    ax.set(xlabel='Residue number', ylabel='pLDDT', ylim=(0, 100), xticks=residues)
    ax.tick_params(axis='x', labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--colabfold-dir', type=Path, default=ROOT / 'results/prediction/colabfold')
    parser.add_argument('--esmfold', type=Path, default=ROOT / 'results/prediction/esmfold_1L2Y.pdb')
    parser.add_argument('--esm-scale', choices=['0-1', '0-100'], required=True, help='Explicit scale of this archived API response; never inferred from its maximum')
    parser.add_argument('--fasta', type=Path, default=ROOT / 'data/prepared/1L2Y.fasta')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/prediction/analysis')
    args = parser.parse_args()
    sequence = fasta_sequence(args.fasta)
    # Validate all inputs before generating any artifacts.
    models = ca_models(args.esmfold, sequence)
    if len(models) != 1:
        raise ValueError('Expected one ESMFold model')
    scores = models[0][2]
    scale_max = 1 if args.esm_scale == '0-1' else 100
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > scale_max)):
        raise ValueError('ESMFold scores are incompatible with explicit scale')
    scores = scores * (100 / scale_max)
    cf_inputs = []
    for path in sorted(args.colabfold_dir.glob('*scores*.json')):
        data = json.loads(path.read_text())
        plddt, pae = validate_scores(data, len(sequence))
        pdb = path.with_name(path.name.replace('_scores_', '_unrelaxed_').replace('.json', '.pdb'))
        if len(ca_models(pdb, sequence)) != 1:
            raise ValueError(f'Expected one ColabFold model: {pdb}')
        cf_inputs.append((path, data, plddt, pae))
    if not cf_inputs:
        raise ValueError('No ColabFold scores found')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_plddt(scores, args.output_dir / 'esmfold_plddt.png')
    summary = {'sequence': sequence, 'ESMFold': {'input_file': str(args.esmfold.relative_to(ROOT)) if args.esmfold.is_relative_to(ROOT) else str(args.esmfold), 'input_scale': args.esm_scale, 'mean_plddt': float(scores.mean()), 'minimum_plddt': float(scores.min()), 'pae': 'not provided by the archived PDB endpoint'}, 'ColabFold': []}
    for path, data, plddt, pae in cf_inputs:
        stem = path.stem
        plot_plddt(plddt, args.output_dir / f'{stem}_plddt.png')
        fig, ax = plt.subplots(figsize=(6.3, 5.4))
        n = len(sequence)
        image = ax.imshow(pae, cmap='viridis', origin='lower', vmin=0, extent=(.5, n+.5, .5, n+.5))
        # AF2 PAE[i,j] is the error of residue j when aligned on residue i.
        # Verified: alphafold/model/modules.py, PredictedAlignedErrorHead.loss:
        # "First num_res are alignment frames, second num_res are the residues."
        ax.set(xlabel='Scored residue (j)', ylabel='Alignment residue (i)', xticks=[1, 5, 10, 15, n], yticks=[1, 5, 10, 15, n])
        fig.colorbar(image, ax=ax, label='PAE (Å)')
        fig.tight_layout()
        fig.savefig(args.output_dir / f'{stem}_pae.png', dpi=200)
        plt.close(fig)
        summary['ColabFold'].append({'scores_file': path.name, 'mean_plddt': float(plddt.mean()), 'mean_pae_angstrom': float(pae.mean()), 'maximum_pae_angstrom': float(pae.max()), 'ptm': float(data['ptm'])})
    summary['interpretation'] = 'Confidence is not measured accuracy. For 20 residues, the pTM length scale is ~0.320 Å; do not interpret its value using general protein thresholds. Compare coordinates with the NMR ensemble separately.'
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
