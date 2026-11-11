#!/usr/bin/env python3
"""Validate three independent MD runs and aggregate trajectory-level observables.

This runs GROMACS analysis only, never mdrun. --single-run is explicitly a
single-trajectory diagnostic and does not certify a completed replicate study.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from prediction_utils import ca_models, fasta_sequence

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261110, 20261111, 20261112)
PAIRS = ((6, 12), (6, 18), (3, 19))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_times(time_ps, total_ns):
    time_ps = np.asarray(time_ps, dtype=float)
    expected = np.arange(0, total_ns * 1000 + 1, 10)
    if time_ps.shape != expected.shape or not np.isfinite(time_ps).all() or not np.allclose(time_ps, expected, atol=.01, rtol=0):
        raise ValueError(f'Incomplete, duplicate or unordered frames: expected {len(expected)} frames, 0–{total_ns*1000} ps every 10 ps')
    return expected


def window_selection(times, start, end, final_ns):
    return (times >= start * 1000) & ((times <= end * 1000) if end == final_ns else (times < end * 1000))


def discover_runs(root, total_ns, seeds=SEEDS):
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('Exactly three distinct predeclared seeds are required')
    runs = []
    for seed in seeds:
        directory = root / f'seed_{seed}'
        needed = ('md.done.json', 'provenance.json', 'md.xtc', 'md.tpr')
        if any(not (directory / name).is_file() for name in needed):
            raise ValueError(f'Missing completed replicate: {directory}')
        done = json.loads((directory / 'md.done.json').read_text())
        provenance = json.loads((directory / 'provenance.json').read_text())
        if done.get('nsteps', 0) < total_ns * 500000 or provenance.get('seed') != seed or provenance.get('target_production_ns', 0) < total_ns:
            raise ValueError(f'Replicate has not completed requested duration: {directory}')
        runs.append((seed, directory, provenance))
    inputs = [r[2].get('input_sha256') for r in runs]
    if not inputs[0] or any(value != inputs[0] for value in inputs[1:]):
        raise ValueError('Replicate source input hashes differ or are missing')
    versions = [r[2].get('gromacs') for r in runs]
    if not versions[0] or len(set(versions)) != 1:
        raise ValueError('Replicate GROMACS versions differ or are missing; analyze platforms separately')
    platforms = [json.dumps({'execution': r[2].get('execution'), 'history': r[2].get('platform_history')}, sort_keys=True) for r in runs]
    if len(set(platforms)) != 1:
        raise ValueError('Replicate execution/platform history differs; do not pool platforms')
    return runs


def nmr_local_reference_distances(path):
    groups = {residue: [] for residue in {r for pair in PAIRS for r in pair}}
    for line in Path(path).read_text().splitlines():
        if line.startswith('ENDMDL'):
            break
        if not line.startswith('ATOM'):
            continue
        residue, atom = int(line[22:26]), line[12:16].strip()
        if residue in groups and not atom.lstrip('0123456789').startswith(('H', 'D')):
            groups[residue].append([float(line[i:i+8]) for i in (30, 38, 46)])
    result = {}
    for a, b in PAIRS:
        if not groups[a] or not groups[b]:
            raise ValueError('Missing NMR reference contact atoms')
        distances = np.linalg.norm(np.array(groups[a])[:, None] - np.array(groups[b])[None], axis=-1)
        result[f'{a}_{b}'] = float(distances.min())
    return result


def read_gro_trajectory(path):
    """Read multi-frame GRO exported by trjconv; coordinates converted nm -> Å."""
    times, coordinates, identities = [], [], None
    with Path(path).open() as handle:
        while True:
            title = handle.readline()
            if not title:
                break
            match = re.search(r'\bt\s*=\s*([-+\d.eE]+)', title)
            if not match:
                raise ValueError('Trajectory GRO frame lacks an explicit time')
            count = int(handle.readline())
            atoms, xyz = [], []
            for _ in range(count):
                line = handle.readline()
                if len(line) < 44:
                    raise ValueError('Truncated GRO atom record')
                atoms.append((int(line[:5]), line[5:10].strip(), line[10:15].strip()))
                xyz.append([float(line[20:28]), float(line[28:36]), float(line[36:44])])
            if not handle.readline().strip():
                raise ValueError('Missing GRO box')
            if identities is None:
                identities = atoms
            elif atoms != identities:
                raise ValueError('Atom identities/order change across trajectory frames')
            times.append(float(match[1]))
            coordinates.append(xyz)
    if not coordinates:
        raise ValueError('No trajectory frames')
    return np.asarray(times), np.asarray(coordinates, dtype=float) * 10, identities


def fitted_coordinates(mobile, reference):
    """Rigid fit without reflection; input can contain many N×3 frames."""
    mobile = np.asarray(mobile, float)
    reference = np.asarray(reference, float)
    centered = mobile - mobile.mean(axis=-2, keepdims=True)
    target = reference - reference.mean(axis=0)
    u, _, vt = np.linalg.svd(np.swapaxes(centered, -1, -2) @ target)
    correction = np.broadcast_to(np.eye(3), u.shape).copy()
    correction[..., -1, -1] = np.linalg.det(u @ vt)
    return centered @ (u @ correction @ vt) + reference.mean(axis=0)


def native_contacts(reference_ca, cutoff):
    i, j = np.triu_indices(len(reference_ca), k=3)
    distances = np.linalg.norm(reference_ca[i] - reference_ca[j], axis=-1)
    selected = distances <= cutoff
    if not selected.any():
        raise ValueError('No native contacts at requested cutoff')
    return i[selected], j[selected], distances[selected]


def geometry_metrics(xyz, identities, reference_ca):
    if not np.isfinite(xyz).all():
        raise ValueError('Nonfinite trajectory coordinates')
    indices = [i for i, atom in enumerate(identities) if atom[2] == 'CA']
    if [identities[i][0] for i in indices] != list(range(1, 21)):
        raise ValueError('Expected one CA for each of residues 1–20')
    ca = xyz[:, indices]
    neighbors = np.linalg.norm(np.diff(ca, axis=1), axis=-1)
    if np.any((neighbors < 2.5) | (neighbors > 4.5)):
        raise ValueError('Protein is broken across PBC or CA connectivity is implausible')
    aligned = fitted_coordinates(ca, reference_ca)
    metrics = {'ca_rmsd_nmr1_angstrom': np.sqrt(np.mean(np.sum((aligned - reference_ca) ** 2, axis=-1), axis=-1))}
    native = {}
    for cutoff in (7, 8, 9):
        i, j, ref = native_contacts(reference_ca, cutoff)
        distances = np.linalg.norm(ca[:, i] - ca[:, j], axis=-1)
        metrics[f'q_ca_{cutoff}A'] = np.mean(distances <= 1.2 * ref, axis=1)
        native[str(cutoff)] = [{'residue_i': int(a+1), 'residue_j': int(b+1), 'reference_distance_angstrom': float(d)} for a, b, d in zip(i, j, ref)]
    for first, second in PAIRS:
        groups = [[i for i, (res, _, name) in enumerate(identities) if res == residue and not name.lstrip('0123456789').startswith('H')] for residue in (first, second)]
        if not all(groups):
            raise ValueError('Missing heavy atoms for local contact')
        distances = np.linalg.norm(xyz[:, groups[0], None, :] - xyz[:, None, groups[1], :], axis=-1)
        metrics[f'heavy_min_{first}_{second}_angstrom'] = distances.min(axis=(1, 2))
    return metrics, aligned, native


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_xvg(path, total_ns, ns=False):
    data = np.loadtxt(path, comments=['#', '@'])
    if data.ndim != 2 or data.shape[1] < 2 or not np.isfinite(data).all():
        raise ValueError(f'Invalid XVG {path}')
    validate_times(data[:, 0] * (1000 if ns else 1), total_ns)
    return data[:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT / 'results/md_replicates_gpu')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gmx', default='gmx')
    parser.add_argument('--total-ns', type=int, default=11)
    parser.add_argument('--keep-coordinates', action='store_true', help='Retain generated multi-frame Protein GRO; otherwise delete after validation')
    parser.add_argument('--common-reference', type=Path, default=ROOT / 'results/md/md.tpr')
    parser.add_argument('--single-run', type=Path, help='Explicit diagnostic only; completion metadata not required')
    parser.add_argument('--trajectory', type=Path, help='Single-run override, e.g. historical md_all_fit_11ns.xtc')
    parser.add_argument('--topology', type=Path, help='Single-run override for md.tpr')
    args = parser.parse_args()
    if args.total_ns != 11:
        parser.error('This prespecified 8–11 ns replicate protocol requires --total-ns 11')
    if (args.trajectory or args.topology) and not args.single_run:
        parser.error('Overrides are reserved for explicit single-trajectory diagnosis')
    runs = [('single_diagnostic', args.single_run, {})] if args.single_run else discover_runs(args.input_root, args.total_ns)
    if not args.common_reference.is_file():
        raise ValueError('Common original production-start TPR is required')
    if not args.single_run:
        for _, _, provenance in runs:
            if provenance['input_sha256'].get('results/md/md.tpr') != sha(args.common_reference):
                raise ValueError('Common reference does not match the declared shared reference')
    output = args.output_dir or ((args.single_run / 'geometry_diagnostic') if args.single_run else (args.input_root / 'aggregate_analysis'))
    if output.exists():
        raise ValueError('Output already exists; use a new analysis output directory')
    sequence = fasta_sequence(ROOT / 'data/prepared/1L2Y.fasta')
    reference_ca = ca_models(ROOT / 'data/raw/1L2Y.pdb', sequence)[0][1]
    output.mkdir(parents=True)
    command_log = []
    gmx_version = subprocess.run([args.gmx, '--version'], capture_output=True, text=True, check=True).stdout
    summaries, residue_rows, all_series, provenance_rows = [], [], {}, []
    windows = [(8, 9), (9, 10), (10, 11), (8, 11)]
    for seed, directory, provenance in runs:
        dest = output / f'seed_{seed}'
        dest.mkdir()
        trajectory = args.trajectory if args.single_run and args.trajectory else directory / 'md.xtc'
        topology = args.topology if args.single_run and args.topology else directory / 'md.tpr'
        hashes = {name: sha(path) for name, path in [('trajectory', trajectory), ('topology', topology)]}
        def run(command, selection):
            result = subprocess.run([args.gmx] + command, input=selection, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=dest)
            command_log.append({'seed': seed, 'argv': [args.gmx] + command, 'stdin': selection, 'returncode': result.returncode})
            with (dest / 'gromacs_analysis.log').open('a') as handle:
                handle.write(result.stdout + '\n')
            if result.returncode:
                raise RuntimeError(f'GROMACS analysis failed: {command}; inspect {dest}')
        whole = dest / 'protein_whole.gro'
        run(['trjconv', '-s', str(topology.resolve()), '-f', str(trajectory.resolve()), '-o', str(whole.resolve()), '-pbc', 'whole', '-b', '0', '-e', '11000'], 'Protein\n')
        time, xyz, identities = read_gro_trajectory(whole)
        time = validate_times(time, args.total_ns)
        from prediction_utils import AA
        observed_sequence = ''.join(AA.get(name, 'X') for _, name, atom in identities if atom == 'CA')
        if observed_sequence != sequence:
            raise ValueError('Trajectory protein sequence differs from the declared reference')
        metrics, aligned, native = geometry_metrics(xyz, identities, reference_ca)
        # These observables use GROMACS; the additional NMR comparison uses CA Kabsch fitting.
        run(['rms', '-s', str(args.common_reference.resolve()), '-f', str(whole.resolve()), '-o', 'rmsd_common_reference.xvg', '-tu', 'ns'], 'Backbone\nBackbone\n')
        run(['gyrate', '-s', str(topology.resolve()), '-f', str(whole.resolve()), '-o', 'gyrate.xvg'], 'Protein\n')
        run(['hbond', '-s', str(topology.resolve()), '-f', str(whole.resolve()), '-num', 'hbond.xvg', '-o', 'hbond.ndx'], 'Protein\nProtein\n')
        metrics.update(backbone_rmsd_common_reference_nm=load_xvg(dest / 'rmsd_common_reference.xvg', 11, ns=True), gyrate_nm=load_xvg(dest / 'gyrate.xvg', 11), protein_internal_hbonds=load_xvg(dest / 'hbond.xvg', 11))
        series_rows = [{'time_ps': int(t), **{name: float(values[i]) for name, values in metrics.items()}} for i, t in enumerate(time)]
        write_csv(dest / 'timeseries.csv', series_rows)
        for start, end in windows:
            selected = window_selection(time, start, end, 11)
            row = {'seed': seed, 'window_ns': f'{start}-{end}', 'frames': int(selected.sum())}
            row.update({name: float(values[selected].mean()) for name, values in metrics.items()})
            summaries.append(row)
            rmsf = np.sqrt(np.mean(np.sum((aligned[selected] - aligned[selected].mean(axis=0)) ** 2, axis=-1), axis=0)) / 10
            residue_rows.extend({'seed': seed, 'window_ns': f'{start}-{end}', 'residue': residue + 1, 'ca_rmsf_nm': float(value)} for residue, value in enumerate(rmsf))
        full_rmsf = np.sqrt(np.mean(np.sum((aligned - aligned.mean(axis=0)) ** 2, axis=-1), axis=0)) / 10
        residue_rows.extend({'seed': seed, 'window_ns': '0-11', 'residue': residue+1, 'ca_rmsf_nm': float(value)} for residue, value in enumerate(full_rmsf))
        if sha(trajectory) != hashes['trajectory'] or sha(topology) != hashes['topology']:
            raise ValueError('Trajectory or topology changed while being analyzed')
        provenance_rows.append({'seed': seed, 'input_directory': str(directory), 'sha256': hashes, 'exported_protein_gro_sha256': sha(whole), 'exported_protein_gro_retained': args.keep_coordinates, 'frames_verified': len(time), 'first_ps': int(time[0]), 'last_ps': int(time[-1])})
        if not args.keep_coordinates:
            whole.unlink()
        all_series[str(seed)] = metrics
    write_csv(output / 'per_trajectory_windows.csv', summaries)
    write_csv(output / 'per_residue_rmsf.csv', residue_rows)
    residue_aggregate = []
    for window in ('0-11', '8-9', '9-10', '10-11', '8-11'):
        for residue in range(1, 21):
            values = [r['ca_rmsf_nm'] for r in residue_rows if r['window_ns'] == window and r['residue'] == residue]
            residue_aggregate.append({'window_ns': window, 'residue': residue, 'n_trajectories': len(values), 'mean_trajectory_ca_rmsf_nm': float(np.mean(values)), 'between_trajectory_sd_nm': float(np.std(values, ddof=1)) if len(values)>1 else None, 'minimum_trajectory_ca_rmsf_nm': min(values), 'maximum_trajectory_ca_rmsf_nm': max(values)})
    write_csv(output / 'residue_trajectory_level_summary.csv', residue_aggregate)
    aggregate = []
    for start, end in windows:
        subset = [r for r in summaries if r['window_ns'] == f'{start}-{end}']
        for metric in all_series[str(runs[0][0])]:
            values = [r[metric] for r in subset]
            aggregate.append({'window_ns': f'{start}-{end}', 'metric': metric, 'n_trajectories': len(values), 'mean_of_trajectory_means': float(np.mean(values)), 'between_trajectory_sd': float(np.std(values, ddof=1)) if len(values)>1 else None, 'minimum_trajectory_mean': min(values), 'maximum_trajectory_mean': max(values)})
    write_csv(output / 'trajectory_level_summary.csv', aggregate)
    metadata = {'status': 'single_trajectory_diagnostic_only' if args.single_run else 'three_completed_replicates_verified', 'n_trajectories': len(runs), 'total_ns_each': 11, 'runs': provenance_rows, 'common_reference_sha256': sha(args.common_reference), 'nmr_reference_sha256': sha(ROOT / 'data/raw/1L2Y.pdb'), 'analysis_gromacs_version': gmx_version, 'q_definition': 'CA pairs with residue separation >=3 and NMR-model1 distance <=7/8/9 Å; contact present if trajectory distance <=1.2 times its reference distance. All atoms are in a whole protein, no minimum-image shortcut.', 'native_contacts': native, 'nmr1_local_reference_distance_angstrom': nmr_local_reference_distances(ROOT / 'data/raw/1L2Y.pdb'), 'local_contact_definition': 'Minimum heavy-atom distance across all atoms of residues W6-P12, W6-P18, Y3-P19; Å. Hydrogens excluded by GRO atom names.', 'rmsf_definition': 'CA fitted to NMR model1 by proper Kabsch rotation and translation; fluctuations about the mean of each indicated window, nm. Interior windows are half-open, final endpoint 11 ns included once.', 'replicate_unit': 'One independently velocity-seeded trajectory; frames are correlated and NMR models are not independent replicates.', 'limitations': 'Three velocity seeds from one minimized structure are not independent initial conformations. Report all three means and their SD/range, no frame-level significance test. Q is a geometric reference-contact fraction, not a thermodynamic folding probability.', 'summary': aggregate}
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'commands.json').write_text(json.dumps(command_log, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for seed, metrics in all_series.items():
        for ax, key, label in zip(axes, ['backbone_rmsd_common_reference_nm', 'q_ca_8A', 'ca_rmsd_nmr1_angstrom'], ['Backbone RMSD / common start (nm)', 'Geometric native-contact fraction Q (8 Å)', 'CA RMSD / NMR model 1 (Å)']):
            ax.plot(time/1000, metrics[key], label=str(seed), linewidth=.8)
            ax.set_ylabel(label)
            ax.axvspan(8, 11, color='gray', alpha=.08)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('Time (ns)')
    fig.tight_layout()
    fig.savefig(output / 'trajectory_comparison.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'status': metadata['status'], 'n_trajectories': len(runs), 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
