"""Verify every planned ablation run and report paired, descriptive results."""
import argparse
import csv
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prediction_utils import ca_models
from prediction_provenance import sha256
from analyze_predictions import validate_scores
from run_prediction_ablation import CONDITIONS, SEEDS, parse_a3m

ROOT = Path(__file__).resolve().parents[1]


def residue_displacements(mobile, reference):
    x, y = mobile - mobile.mean(0), reference - reference.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return np.linalg.norm(x @ u @ correction @ vt - y, axis=1)


def paired_deltas(rows, treatment, reference):
    indexed = {}
    for row in rows:
        key = (row['condition'], row['seed'])
        if key in indexed:
            raise ValueError(f'Duplicate condition/seed: {key}')
        indexed[key] = row
    if {s for c, s in indexed if c == treatment} != set(SEEDS) or {s for c, s in indexed if c == reference} != set(SEEDS):
        raise ValueError('Paired comparison requires all prespecified seeds in both conditions')
    return [{'seed': s, 'delta_median_ca_rmsd_angstrom': indexed[treatment, s]['median_ca_rmsd_angstrom'] - indexed[reference, s]['median_ca_rmsd_angstrom'], 'delta_mean_plddt': indexed[treatment, s]['mean_plddt'] - indexed[reference, s]['mean_plddt']} for s in SEEDS]


def verify_run(experiment, plan, condition, seed):
    directory = experiment / 'runs' / condition / f'seed_{seed}'
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['condition'] != condition or manifest['seed'] != seed or manifest['plan_sha256'] != sha256(experiment / 'plan.json') or manifest['weights_sha256'] != plan['weights_model_1_ptm_sha256']:
        raise ValueError('Manifest identity/plan/weights mismatch')
    if manifest['started_utc'] <= plan['created_before_inference_utc']:
        raise ValueError('Plan was not retained before the run')
    msa = experiment / plan['inputs'][condition]['path']
    if sha256(msa) != manifest['input_sha256'] or manifest['input_sha256'] != plan['inputs'][condition]['sha256']:
        raise ValueError('MSA input changed')
    records = parse_a3m(msa.read_text(), plan['sequence'])
    if len(records) != plan['inputs'][condition]['rows_including_query']:
        raise ValueError('Input depth mismatch')
    current = {p.name: sha256(p) for p in directory.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if current != manifest['output_sha256']:
        raise ValueError(f'Output inventory/hash mismatch in {directory}')
    log = (directory / 'run.log').read_text()
    config = json.loads((directory / 'config.json').read_text())
    expected = {'random_seed': seed, 'num_models': 1, 'num_seeds': 1, 'num_recycles': 3, 'model_type': 'alphafold2_ptm', 'model_order': [1], 'use_templates': False, 'num_relax': 0}
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f'Unexpected config {key}: {config[key]}')
    if [int(v) for v in re.findall(r'recycle=(\d+)', log)] != [0, 1, 2, 3] or 'Skipping ' in log:
        raise ValueError('Run did not complete exactly one inference with recycle 0-3')
    if 'SUBMIT:' in log:
        raise ValueError('Unexpected online MSA submission')
    version = re.search(r'Running colabfold (\S+)', log)
    if not version or version[1] != config['version']:
        raise ValueError('Version mismatch')
    sizes = re.search(r'Setting max_seq=(\d+), max_extra_seq=(\d+)', log)
    if not sizes:
        raise ValueError('Missing effective MSA capacity evidence')
    scores_paths, pdb_paths = list(directory.glob('*scores*.json')), list(directory.glob('*unrelaxed*.pdb'))
    if len(scores_paths) != 1 or len(pdb_paths) != 1:
        raise ValueError('Expected one prediction per run')
    scores = json.loads(scores_paths[0].read_text())
    plddt, pae = validate_scores(scores, len(plan['sequence']))
    models = ca_models(pdb_paths[0], plan['sequence'])
    if len(models) != 1 or not np.allclose(models[0][2], plddt, atol=.011, rtol=0):
        raise ValueError('Coordinate model and confidence are inconsistent')
    return models[0][1], plddt, pae, scores['ptm'], {'effective_max_seq': int(sizes[1]), 'effective_max_extra_seq': int(sizes[2]), 'colabfold_version': config['version'], 'msa_mode_config': config['msa_mode'], 'query_only_is_no_external_homolog': condition == 'query_only'}


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, default=ROOT / 'results/prediction_ablation')
    args = parser.parse_args()
    base = args.experiment_dir
    plan = json.loads((base / 'plan.json').read_text())
    if plan['seeds_in_execution_order'] != SEEDS or plan['conditions_in_execution_order'] != CONDITIONS:
        raise ValueError('Unexpected experiment plan')
    nmr = ca_models(ROOT / 'data/raw/1L2Y.pdb', plan['sequence'])
    if len(nmr) != 38:
        raise ValueError('Expected the complete 38-model NMR ensemble')
    interior, terminal = np.array(plan['interior_residues']) - 1, np.array(plan['terminal_residues']) - 1
    rows, per_model, per_residue = [], [], []
    residue_errors, confidences = {}, {}
    for condition in CONDITIONS:
        residue_errors[condition], confidences[condition] = [], []
        for seed in SEEDS:
            coords, plddt, pae, ptm, effective = verify_run(base, plan, condition, seed)
            errors = np.array([residue_displacements(coords, reference) for _, reference, _ in nmr])
            full = np.sqrt(np.mean(errors ** 2, axis=1))
            interior_rmsd = np.sqrt(np.mean(errors[:, interior] ** 2, axis=1))
            terminal_rmsd = np.sqrt(np.mean(errors[:, terminal] ** 2, axis=1))
            median_error = np.median(errors, axis=0)
            row = {'condition': condition, 'seed': seed, 'mean_plddt': float(plddt.mean()), 'mean_pae_angstrom': float(pae.mean()), 'maximum_pae_angstrom': float(pae.max()), 'ptm_json_rounded': ptm, 'model1_ca_rmsd_angstrom': float(full[0]), 'minimum_ca_rmsd_angstrom': float(full.min()), 'median_ca_rmsd_angstrom': float(np.median(full)), 'maximum_ca_rmsd_angstrom': float(full.max()), 'median_interior_global_fit_rmsd_angstrom': float(np.median(interior_rmsd)), 'median_terminal_global_fit_rmsd_angstrom': float(np.median(terminal_rmsd)), 'interior_mean_plddt': float(plddt[interior].mean()), 'terminal_mean_plddt': float(plddt[terminal].mean()), 'residue_plddt_error_pearson_descriptive': float(np.corrcoef(plddt, median_error)[0, 1]), **effective}
            rows.append(row)
            for index, (model, _, _) in enumerate(nmr):
                per_model.append({'condition': condition, 'seed': seed, 'nmr_model': model, 'ca_rmsd_angstrom': full[index], 'interior_global_fit_rmsd_angstrom': interior_rmsd[index], 'terminal_global_fit_rmsd_angstrom': terminal_rmsd[index]})
                for residue in range(len(plddt)):
                    per_residue.append({'condition': condition, 'seed': seed, 'nmr_model': model, 'residue': residue + 1, 'global_fit_ca_displacement_angstrom': errors[index, residue], 'plddt': plddt[residue], 'subset': 'interior_3_18' if residue in interior else 'terminal_1_2_19_20'})
            residue_errors[condition].append(median_error)
            confidences[condition].append(plddt)
    if len(set(row['colabfold_version'] for row in rows)) != 1:
        raise ValueError('Software versions differ across runs')
    contrasts = {f'{a}_minus_{b}': paired_deltas(rows, a, b) for a, b in [('deduplicated_saved_msa', 'query_only'), ('raw_saved_msa', 'deduplicated_saved_msa')]}
    aggregates = {condition: {'mean_of_seed_median_rmsd_angstrom': float(np.mean([r['median_ca_rmsd_angstrom'] for r in rows if r['condition'] == condition])), 'range_of_seed_median_rmsd_angstrom': [float(f([r['median_ca_rmsd_angstrom'] for r in rows if r['condition'] == condition])) for f in (min, max)], 'mean_plddt_over_seeds': float(np.mean([r['mean_plddt'] for r in rows if r['condition'] == condition]))} for condition in CONDITIONS}
    summary = {'all_nine_runs_verified': True, 'plan_sha256': sha256(base / 'plan.json'), 'reference_nmr_sha256': sha256(ROOT / 'data/raw/1L2Y.pdb'), 'reference_models': len(nmr), 'rows': rows, 'condition_descriptive_summary': aggregates, 'paired_contrasts': contrasts, 'interpretation_limits': plan['analysis_limits'] + ' max_seq/max_extra_seq are effective capacity settings, not Neff; input unique aligned counts are reported separately. pLDDT-error correlations and condition shifts here are descriptive; no causal or population claim follows. Query-only still uses the same MSA-capable AlphaFold2 model and is not ESMFold. Raw and deduplicated inputs share the same two unique aligned sequences; their different row layout and effective capacity can alter stochastic feature processing, so differences cannot be attributed solely to additional evolutionary information. This historical peptide has not been established as held out from model training.'}
    write_csv(base / 'summary.csv', rows)
    write_csv(base / 'per_nmr_model.csv', per_model)
    write_csv(base / 'per_residue.csv', per_residue)
    (base / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.3))
    names = ['Query only', 'Deduplicated\nsaved MSA', 'Raw saved\nMSA']
    colors = ['#176b87', '#b15c23', '#6848a1']
    for seed in SEEDS:
        subset = [next(r for r in rows if r['condition'] == c and r['seed'] == seed) for c in CONDITIONS]
        axes[0, 0].plot(range(3), [r['median_ca_rmsd_angstrom'] for r in subset], marker='o', label=f'seed {seed}')
    axes[0, 0].set(xticks=range(3), xticklabels=names, ylabel='Median Cα RMSD over NMR models (Å)', title='A  Paired seeds; all 38 NMR models retained')
    axes[0, 0].legend(fontsize=8)
    for condition, color, label in zip(CONDITIONS, colors, names):
        subset = [r for r in rows if r['condition'] == condition]
        axes[0, 1].scatter([r['mean_plddt'] for r in subset], [r['median_ca_rmsd_angstrom'] for r in subset], color=color, label=label.replace('\n', ' '), s=42)
        axes[1, 0].plot(range(1, 21), np.median(residue_errors[condition], axis=0), label=label.replace('\n', ' '), color=color)
        axes[1, 1].plot(range(1, 21), np.median(confidences[condition], axis=0), label=label.replace('\n', ' '), color=color)
    axes[0, 1].set(xlabel='Mean pLDDT', ylabel='Median Cα RMSD (Å)', title='B  Confidence versus measured structural difference')
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].set(ylabel='Median fitted CA displacement (Å)', title='C  Per-residue difference: median NMR, then seed')
    axes[1, 1].set(ylabel='Median pLDDT across three seeds', title='D  Per-residue confidence')
    for ax in axes[1]:
        ax.set(xlabel='Residue number', xticks=[1, 2, 5, 10, 15, 19, 20])
        ax.axvspan(.5, 2.5, color='gray', alpha=.13)
        ax.axvspan(18.5, 20.5, color='gray', alpha=.13)
    axes[1, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(base / 'ablation_overview.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'condition_summary': aggregates, 'paired_contrasts': contrasts}, indent=2))


if __name__ == '__main__':
    main()
