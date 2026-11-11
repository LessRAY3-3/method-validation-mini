"""Package local MD evidence without uploading it; explicit file allowlists only."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261110, 20261111, 20261112)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def study_files(study):
    if study == 'historical':
        files = [ROOT / 'results/md' / (base + ext)
                 for base in ('md', 'md_extended', 'md_extended2')
                 for ext in ('.tpr', '.xtc', '.cpt')]
        files += [ROOT / 'results/md/topol.top', ROOT / 'results/md/posre.itp']
        files += sorted((ROOT / 'config').glob('*.mdp'))
        files += sorted((ROOT / 'report/evidence').glob('*.txt'))
        return files, ROOT / 'results/validation/evidence_manifest.json', 'md-evidence.tar.gz'
    base = ROOT / 'results/md_replicates_gpu'
    files = [base / 'queue_status.json', base / 'local_transfer_verification.json']
    for seed in SEEDS:
        directory = base / f'seed_{seed}'
        for stage in ('nvt', 'npt', 'md'):
            files += [directory / (stage + ext) for ext in
                      ('.tpr', '.xtc', '.cpt', '.edr', '.log', '.gro', '.mdp', '_processed.mdp', '.done.json')]
        files += [directory / name for name in ('minim.gro', 'topol.top', 'posre.itp', 'provenance.json', 'commands.jsonl', 'workflow.log')]
    files += [ROOT / 'results/md' / name for name in ('md.tpr', 'minim.gro', 'topol.top', 'posre.itp')]
    files += [ROOT / 'config' / name for name in ('nvt.mdp', 'npt.mdp', 'md.mdp', 'experiments/md_gpu_formal_protocol.json')]
    files += [ROOT / 'data' / name for name in ('raw/1L2Y.pdb', 'prepared/1L2Y_model1.pdb', 'prepared/1L2Y.fasta')]
    files += [ROOT / 'scripts' / name for name in ('run_md_replicate.py', 'run_gpu_formal_queue.py', 'analyze_md_replicates.py', 'prediction_utils.py', 'smoke_gpu.sh', 'package_evidence.py')]
    files += [base / 'setup' / name for name in ('environment_explicit.txt', 'gromacs_version.txt', 'execution_hashes.txt', 'first_production_snapshot.json', 'nvidia_smi_initial.txt', 'original_tpr_read.log', 'smoke_console.log', 'forcefield_sha256.txt', 'forcefield_comparison.json')]
    # Frozen original remote analysis; later audits can be separate artifacts.
    aggregate = base / 'aggregate_analysis'
    files += [aggregate / name for name in ('summary.json', 'commands.json', 'per_trajectory_windows.csv', 'trajectory_level_summary.csv', 'per_residue_rmsf.csv', 'residue_trajectory_level_summary.csv', 'trajectory_comparison.png')]
    for seed in SEEDS:
        files += [aggregate / f'seed_{seed}' / name for name in ('timeseries.csv', 'rmsd_common_reference.xvg', 'gyrate.xvg', 'hbond.xvg', 'hbond.ndx', 'gromacs_analysis.log')]
    return files, ROOT / 'results/validation/gpu_replicates_evidence_manifest.json', 'md-gpu-replicates-evidence.tar.gz'


def validate_gpu():
    base = ROOT / 'results/md_replicates_gpu'
    queue = json.loads((base / 'queue_status.json').read_text())
    summary = json.loads((base / 'aggregate_analysis/summary.json').read_text())
    if queue.get('status') != 'complete' or set(queue.get('completed_seeds', [])) != set(SEEDS):
        raise ValueError('Formal queue is not complete')
    if summary.get('status') != 'three_completed_replicates_verified' or summary.get('n_trajectories') != 3:
        raise ValueError('Missing successful original three-replicate analysis')
    if {run['seed'] for run in summary['runs']} != set(SEEDS):
        raise ValueError('Wrong replicate seeds in analysis')
    for run in summary['runs']:
        if run['frames_verified'] != 1101:
            raise ValueError('Incomplete frame count')
        for label, filename in (('trajectory', 'md.xtc'), ('topology', 'md.tpr')):
            if sha(base / f"seed_{run['seed']}" / filename) != run['sha256'][label]:
                raise ValueError('Transferred trajectory or TPR differs from analyzed data')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--study', choices=('historical', 'gpu-replicates'), default='historical')
    cli.add_argument('--output', type=Path)
    args = cli.parse_args()
    files, target, name = study_files(args.study)
    files = sorted(set(files))
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        raise SystemExit('Missing required local evidence: ' + ', '.join(missing))
    if any(p.is_symlink() or not p.resolve().is_relative_to(ROOT) for p in files):
        raise ValueError('Evidence must be regular project files, not symlinks')
    if args.study == 'gpu-replicates':
        validate_gpu()
    output = args.output or ROOT / 'artifacts' / name
    manifest = {'status': 'local archive; not uploaded', 'study': args.study,
                'scope': 'Explicit raw-data allowlist and original analysis. Script hashes describe the packaging-time checkout; execution hashes record original remote scripts. Later audit outputs are separate and are not silently substituted.',
                'files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in files + [target]:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    # Read the compressed archive, not just source files, to verify every member.
    expected = {row['path']: row['sha256'] for row in manifest['files']}
    expected[str(target.relative_to(ROOT))] = sha(target)
    with tarfile.open(temporary, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('Archive membership differs from manifest')
        for member in members:
            if not member.isfile():
                raise ValueError('Archive contains a non-regular file')
            with archive.extractfile(member) as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != expected[member.name]:
                    raise ValueError('Archive hash mismatch or source changed during packaging: ' + member.name)
    temporary.replace(output)
    archive_hash = sha(output)
    output.with_suffix(output.suffix + '.sha256').write_text(f'{archive_hash}  {output.name}\n')
    verification = {'status': 'archive_members_and_hashes_verified', 'study': args.study, 'archive': str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output), 'archive_bytes': output.stat().st_size, 'archive_sha256': archive_hash, 'verified_members_including_manifest': len(expected), 'uploaded': False}
    target.with_name(target.stem.replace('_manifest', '_archive_verification') + '.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
