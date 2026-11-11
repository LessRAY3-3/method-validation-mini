#!/usr/bin/env python3
"""Independent velocity replicate from the shared minimized coordinates.

Resume with the same seed; increase --total-ns to extend from checkpoint.
No original MD files are changed. Stage markers are written only after success.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import platform
import signal
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--total-ns', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2, help='OpenMP threads; one thread-MPI rank')
    parser.add_argument('--gmx', default=os.environ.get('GMX', 'gmx'))
    parser.add_argument('--execution', choices=('cpu', 'auto', 'cuda'), default='cpu')
    parser.add_argument('--gpu-id', help='Optional GROMACS GPU device IDs, e.g. 0')
    parser.add_argument('--output-root', default='results/md_replicates', help='Run output root, relative to project unless absolute')
    parser.add_argument('--run-label', default='', help='New directory for formal runs, e.g. ubuntu_cuda; empty resumes the original pilot')
    parser.add_argument('--allow-platform-change', action='store_true', help='Explicitly allow checkpoint continuation on a different platform; numerical identity is not guaranteed')
    args = parser.parse_args()
    if args.total_ns < 1 or args.threads < 1:
        parser.error('Use a positive duration and positive thread count')
    if args.run_label and not re.fullmatch(r'[a-zA-Z0-9_-]+', args.run_label):
        parser.error('run-label may contain only letters, numbers, underscores and hyphens')
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out = output_root / args.run_label / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    lock_handle = (out / '.runner.lock').open('a')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This run directory already has an active driver')
    gmx = args.gmx
    version = subprocess.run([gmx, '--version'], capture_output=True, text=True, check=True).stdout
    if args.execution == 'cuda' and not re.search(r'GPU support:\s*CUDA', version, re.I):
        parser.error('CUDA execution requested but this GROMACS build does not report CUDA support')
    if args.gpu_id and args.execution == 'cpu':
        parser.error('--gpu-id requires auto or cuda execution')
    current_platform = {'system': platform.system(), 'machine': platform.machine(), 'gromacs': version}
    platform_history = []
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [
        ROOT / 'results/md/minim.gro', ROOT / 'results/md/topol.top',
        ROOT / 'results/md/posre.itp', ROOT / 'results/md/md.tpr',
        ROOT / 'config/nvt.mdp', ROOT / 'config/npt.mdp', ROOT / 'config/md.mdp']}
    manifest = out / 'provenance.json'
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        platform_history = prior.get('platform_history', [])
        if platform_history and platform_history[-1] != current_platform:
            if not args.allow_platform_change:
                raise RuntimeError('Platform changed. Prefer a new --run-label from minim, or explicitly use --allow-platform-change for non-bitwise checkpoint continuation')
        if prior['input_sha256'] != inputs:
            raise RuntimeError('Original input hashes changed; use a new run directory')
    for name in ('minim.gro', 'topol.top', 'posre.itp'):
        dest = out / name
        if not dest.exists():
            shutil.copy2(ROOT / 'results/md' / name, dest)
    if not platform_history or platform_history[-1] != current_platform:
        platform_history.append(current_platform)
    metadata = dict(seed=args.seed, target_production_ns=args.total_ns, threads=args.threads,
                    input_sha256=inputs, gromacs=version, platform_history=platform_history, execution=args.execution, gpu_id=args.gpu_id, run_label=args.run_label or ('local_pilot' if output_root.name == 'md_replicates' else output_root.name),
                    design='Independent initial velocities from the same minimized conformation; 100 ps restrained NVT + 100 ps restrained NPT; not independent initial conformations',
                    analysis_reference='Original results/md/md.tpr: shared original production-start structure, not each replicate NPT endpoint')
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')

    active_child = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        if active_child is not None and active_child.poll() is None:
            active_child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def run(argv, stdin=None):
        nonlocal active_child
        if stop_requested:
            raise SystemExit('Stopped; resume from the saved checkpoint')
        with (out / 'commands.jsonl').open('a') as handle:
            handle.write(json.dumps({'time_unix': time.time(), 'cwd': str(out), 'argv': argv, 'stdin': stdin}) + '\n')
        with (out / 'workflow.log').open('a') as handle:
            active_child = subprocess.Popen(argv, cwd=out, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL, text=True, stdout=handle, stderr=subprocess.STDOUT)
            active_child.communicate(stdin)
            code = active_child.returncode
            active_child = None
            if stop_requested:
                raise SystemExit('Stopped; completed stages retained, current stage resumes from checkpoint')
            if code:
                raise subprocess.CalledProcessError(code, argv)

    for stage in ('nvt', 'npt', 'md'):
        done = out / f'{stage}.done.json'
        steps = args.total_ns * 500000 if stage == 'md' else 50000
        if done.exists() and json.loads(done.read_text())['nsteps'] >= steps:
            print(f'{args.seed}: {stage} already complete', flush=True)
            continue
        mdp = (ROOT / 'config' / f'{stage}.mdp').read_text()
        mdp = re.sub(r'(ld-seed\s*=\s*)\d+', rf'\g<1>{args.seed}', mdp)
        mdp = re.sub(r'(gen_seed\s*=\s*)\d+', rf'\g<1>{args.seed}', mdp)
        mdp = re.sub(r'(nsteps\s*=\s*)\d+', rf'\g<1>{steps}', mdp)
        (out / f'{stage}.mdp').write_text(mdp)
        tpr = out / f'{stage}.tpr'
        if stage == 'md' and tpr.exists():
            # Conversion changes only the end time; velocities/RNG resume from checkpoint.
            run([gmx, 'convert-tpr', '-s', str(tpr), '-until', str(args.total_ns * 1000), '-o', str(tpr)])
        elif not tpr.exists():
            previous = {'nvt': 'minim', 'npt': 'nvt', 'md': 'npt'}[stage]
            cmd = [gmx, 'grompp', '-f', f'{stage}.mdp', '-c', f'{previous}.gro', '-p', 'topol.top', '-o', f'{stage}.tpr', '-po', f'{stage}_processed.mdp']
            if stage != 'nvt':
                cmd += ['-t', f'{previous}.cpt']
            if stage != 'md':
                cmd += ['-r', f'{previous}.gro']
            run(cmd)
        print(f'{args.seed}: running {stage}, {steps} steps, {args.threads} threads', flush=True)
        cmd = [gmx, 'mdrun', '-s', f'{stage}.tpr', '-deffnm', stage, '-ntmpi', '1', '-ntomp', str(args.threads)]
        # Let GROMACS choose compatible PME/update/bonded assignments on CUDA.
        if args.execution == 'cpu':
            cmd += ['-nb', 'cpu', '-pme', 'cpu']
        else:
            cmd += ['-nb', 'auto', '-pme', 'auto']
        if args.gpu_id:
            cmd += ['-gpu_id', args.gpu_id]
        if (out / f'{stage}.cpt').exists():
            cmd += ['-cpi', f'{stage}.cpt', '-append']
        run(cmd)
        checkpoint = subprocess.run([gmx, 'dump', '-cp', str(out / f'{stage}.cpt')], capture_output=True, text=True, check=True).stdout
        match = re.search(r'^step\s*=\s*(\d+)', checkpoint, re.M)
        if not match or int(match.group(1)) < steps:
            raise SystemExit(f'{stage} stopped before requested final step; checkpoint retained, stage NOT marked complete')
        done.write_text(json.dumps({'nsteps': steps, 'completed_unix': time.time()}) + '\n')
        print(f'{args.seed}: finished {stage}', flush=True)

    reference = str(ROOT / 'results/md/md.tpr')
    run([gmx, 'trjconv', '-s', reference, '-f', 'md.xtc', '-o', 'md_centered.xtc', '-pbc', 'mol', '-center'], '1\n0\n')
    run([gmx, 'trjconv', '-s', reference, '-f', 'md_centered.xtc', '-o', 'md_fit.xtc', '-fit', 'rot+trans'], '4\n0\n')
    run([gmx, 'rms', '-s', reference, '-f', 'md_fit.xtc', '-o', 'rmsd_common_reference.xvg', '-tu', 'ns'], '4\n4\n')
    run([gmx, 'rmsf', '-s', reference, '-f', 'md_fit.xtc', '-o', 'rmsf_ca.xvg', '-res', '-fit'], '3\n')
    run([gmx, 'gyrate', '-s', reference, '-f', 'md_fit.xtc', '-o', 'gyrate.xvg'], '1\n')
    run([gmx, 'hbond', '-s', reference, '-f', 'md_fit.xtc', '-num', 'hbond.xvg', '-o', 'hbond.ndx'], '1\n1\n')
    run([gmx, 'energy', '-f', 'md.edr', '-o', 'temperature_pressure.xvg'], 'Temperature\nPressure\n\n')
    run([gmx, 'check', '-f', 'md.xtc'])
    print(f'{args.seed}: analysis complete', flush=True)


if __name__ == '__main__':
    main()
