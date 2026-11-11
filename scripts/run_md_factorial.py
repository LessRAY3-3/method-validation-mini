#!/usr/bin/env python3
"""Isolated, resumable B/C/D branches of the fixed starting-structure/NPT study.

B extends existing A NPT; C starts from prepared model24; D extends C NPT.
Every production run uses its OWN topology, and starts at production time zero.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20261113, 20261114, 20261115]
NPT_PS = {'B': 1000, 'C': 100, 'D': 1000}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def validate_protocol(root, path):
    p = json.loads(path.read_text())
    if (p.get('seeds') != SEEDS or p.get('output_root') != 'results/md_factorial'
            or p.get('production_ns_each') != 11 or p.get('nvt_ps') != 100
            or p.get('npt_ps_by_cell') != NPT_PS):
        raise ValueError('Protocol differs from fixed factorial design')
    for field in ('input_sha256', 'source_sha256'):
        if not p.get(field):
            raise ValueError('Missing frozen ' + field)
        for name, want in p[field].items():
            if sha(root / name) != want:
                raise ValueError('Frozen file changed: ' + name)
    if not p.get('gromacs_binary_sha256'):
        raise ValueError('Missing GROMACS binary hash')
    return p


def stage_mdp(text, seed, steps):
    for key, val in [('nsteps', steps), ('ld-seed', seed), ('gen_seed', seed)]:
        text = re.sub(r'(?m)^(\s*' + re.escape(key) + r'\s*=\s*)\S+',
                      lambda m: m[1] + str(val), text)
    return text


def check_stage_text(log, checkpoint, steps):
    if re.search(r'LINCS WARNING|Fatal error', log, re.I):
        raise ValueError('Numerical warning/error in stage log')
    for phrase in ('short-ranged interactions on the GPU', 'PME tasks will do all aspects on the GPU'):
        if phrase not in log:
            raise ValueError('GPU offload not confirmed')
    match = re.search(r'^step\s*=\s*(\d+)', checkpoint, re.M)
    if not match or int(match[1]) != steps:
        raise ValueError('Checkpoint is not at the exact requested final step')


def check_tpr(dump, stage, steps):
    def number(key):
        match = re.search(r'^\s*' + re.escape(key) + r'\s*=\s*(\S+)', dump, re.M)
        return float(match[1]) if match else None
    if number('nsteps') != steps or number('dt') != .002:
        raise ValueError('Actual TPR duration/timestep differs from protocol')
    has_posres = bool(re.search(r'functype\[\d+\]=POSRES,', dump))
    if has_posres != (stage != 'md'):
        raise ValueError('Actual TPR position restraints differ from stage design')
    if stage == 'md' and (number('tinit') != 0 or number('init-step') != 0):
        raise ValueError('Production TPR must start its clock at zero')


def source_files(source, branch):
    required = ['minim.gro', 'topol.top', 'posre.itp']
    if branch:
        required += ['nvt.gro', 'nvt.cpt', 'nvt.tpr', 'nvt.log',
                     'npt.gro', 'npt.cpt', 'npt.tpr', 'npt.log', 'npt.edr', 'npt.xtc']
        marker = source / 'npt.done.json'
        if not marker.exists() or json.loads(marker.read_text())['nsteps'] != 50000:
            raise ValueError('Branch source must have completed exactly 100 ps NPT')
    for name in required:
        if not (source / name).is_file():
            raise ValueError('Missing branch/preparation input: ' + str(source / name))
    # Preserve every NPT checkpoint-linked output for GROMACS append checksum checks.
    names = set(required)
    if branch:
        names.update(p.name for p in source.glob('npt.*') if p.is_file() and p.name != 'npt.done.json')
        names.update(p.name for p in source.glob('nvt*') if p.is_file() and p.name != 'nvt.done.json')
    names.update(p.name for p in source.glob('*.itp') if p.is_file())
    return sorted(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--gmx', required=True)
    ap.add_argument('--cell', choices=NPT_PS, required=True)
    ap.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = ap.parse_args()
    protocol = args.protocol.resolve()
    p = validate_protocol(ROOT, protocol)
    gmx = str(Path(args.gmx).resolve())
    if sha(gmx) != p['gromacs_binary_sha256']:
        raise ValueError('GROMACS binary differs from frozen protocol')
    version = subprocess.run([gmx, '--version'], capture_output=True, text=True, check=True).stdout
    if not re.search(r'GPU support:\s*CUDA', version, re.I):
        raise ValueError('CUDA GROMACS required')
    out = ROOT / p['output_root'] / args.cell / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = (ROOT / 'setup/md_factorial/model24' if args.cell == 'C' else
              ROOT / ('results/md_replicates_gpu_confirmation' if args.cell == 'B' else
                      'results/md_factorial/C') / f'seed_{args.seed}')
    names = source_files(source, args.cell != 'C')
    origin = {name: sha(source / name) for name in names}
    identity = {'cell': args.cell, 'seed': args.seed, 'protocol_sha256': sha(protocol),
                'source_directory': str(source.relative_to(ROOT)), 'source_sha256': origin,
                'gromacs_binary_sha256': sha(gmx)}
    manifest = out / 'provenance.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['identity'] != identity:
            raise ValueError('Resume inputs, branch source, or protocol changed')
    else:
        save(manifest, {'identity': identity, 'gromacs': version, 'created_unix': time.time(),
                        'seed': args.seed, 'target_production_ns': 11,
                        'production_ns_each': 11, 'nvt_ps': 100, 'npt_ps': NPT_PS[args.cell],
                        'analysis_reference': 'own md.tpr; cross-condition reference analysis is separate'})
    completion = out / 'complete.json'
    if completion.exists():
        prior = json.loads(completion.read_text())
        for name, want in prior['output_sha256'].items():
            if sha(out / name) != want:
                raise ValueError('Completed output changed: ' + name)
        print(f'{args.cell}/{args.seed}: already complete and verified', flush=True)
        return
    # Transactional copy: resume a failed copy only BEFORE dynamics began.
    copied = out / 'copy.done.json'
    if not copied.exists():
        for name in names:
            shutil.copy2(source / name, out / name)
            if sha(out / name) != origin[name]:
                raise ValueError('Copy checksum mismatch: ' + name)
        save(copied, origin)

    child = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None:
            child.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def run(argv, stdin=None, capture=False):
        nonlocal child
        if stopping:
            raise RuntimeError('Stopped; checkpoints retained')
        if sha(protocol) != identity['protocol_sha256']:
            raise ValueError('Protocol changed while running')
        validate_protocol(ROOT, protocol)
        with (out / 'commands.jsonl').open('a') as f:
            f.write(json.dumps({'time_unix': time.time(), 'argv': argv, 'stdin': stdin}) + '\n')
        with (out / 'workflow.log').open('a') as log:
            child = subprocess.Popen(argv, cwd=out, stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                                     stdout=subprocess.PIPE if capture else log, stderr=log, text=True)
            stdout, _ = child.communicate(stdin)
            code = child.returncode
            child = None
        if stopping or code:
            raise RuntimeError(f'Command failed/stopped ({code}): {argv}')
        return stdout or ''

    def check(stage, steps):
        cp = run([gmx, 'dump', '-cp', f'{stage}.cpt'], capture=True)
        check_stage_text((out / f'{stage}.log').read_text(), cp, steps)

    for stage in ('nvt', 'npt', 'md'):
        steps = {'nvt': 50000, 'npt': NPT_PS[args.cell] * 500, 'md': 5500000}[stage]
        done = out / f'{stage}.done.json'
        if stage == 'nvt' and args.cell != 'C':
            check(stage, steps)
            save(done, {'nsteps': steps, 'inherited_from': identity['source_directory']})
            continue
        if done.exists():
            if json.loads(done.read_text())['nsteps'] != steps:
                raise ValueError('Stage duration differs from frozen design')
            check(stage, steps)
            continue
        mdp = stage_mdp((ROOT / 'config' / f'{stage}.mdp').read_text(), args.seed, steps)
        if stage == 'md' and not (re.search(r'(?m)^gen_vel\s*=\s*no\s*$', mdp) and re.search(r'(?m)^continuation\s*=\s*yes\s*$', mdp)):
            raise ValueError('Production must retain velocities and continue constraints')
        (out / f'{stage}.mdp').write_text(mdp)
        if stage == 'npt' and args.cell != 'C':
            # A separate TPR avoids replacing a copied TPR until conversion succeeds.
            converted = out / 'npt_extension.done.json'
            if not converted.exists():
                run([gmx, 'convert-tpr', '-s', 'npt.tpr', '-until', '1000', '-o', 'npt_long.tpr'])
                (out / 'npt_long.tpr').replace(out / 'npt.tpr')
                save(converted, {'nsteps': steps, 'source_checkpoint_sha256': origin['npt.cpt']})
        elif not (out / f'{stage}.tpr').exists():
            prev = {'nvt': 'minim', 'npt': 'nvt', 'md': 'npt'}[stage]
            cmd = [gmx, 'grompp', '-f', f'{stage}.mdp', '-c', f'{prev}.gro', '-p', 'topol.top',
                   '-o', f'{stage}.tpr', '-po', f'{stage}_processed.mdp']
            if stage != 'nvt':
                cmd += ['-t', f'{prev}.cpt']
            if stage != 'md':
                cmd += ['-r', f'{prev}.gro']
            run(cmd)
        dump = run([gmx, 'dump', '-s', f'{stage}.tpr'], capture=True)
        (out / f'{stage}_tpr_dump.txt').write_text(dump)
        check_tpr(dump, stage, steps)
        print(f'{args.cell}/{args.seed}: {stage} {steps} steps', flush=True)
        cmd = [gmx, 'mdrun', '-s', f'{stage}.tpr', '-deffnm', stage, '-ntmpi', '1',
               '-ntomp', str(p.get('threads', 4)), '-nb', 'gpu', '-pme', 'gpu',
               '-gpu_id', str(p.get('gpu_id', '0'))]
        if (out / f'{stage}.cpt').exists():
            cmd += ['-cpi', f'{stage}.cpt', '-append']
        run(cmd)
        check(stage, steps)
        save(done, {'nsteps': steps, 'completed_unix': time.time(),
                    'tpr_sha256': sha(out / f'{stage}.tpr'), 'checkpoint_sha256': sha(out / f'{stage}.cpt')})

    # Own topology is mandatory: model24 solvent counts may differ from model1.
    run([gmx, 'trjconv', '-s', 'md.tpr', '-f', 'md.xtc', '-o', 'md_centered.xtc', '-pbc', 'mol', '-center'], '1\n0\n')
    run([gmx, 'trjconv', '-s', 'md.tpr', '-f', 'md_centered.xtc', '-o', 'md_fit.xtc', '-fit', 'rot+trans'], '4\n0\n')
    run([gmx, 'rms', '-s', 'md.tpr', '-f', 'md_fit.xtc', '-o', 'rmsd_own_reference.xvg', '-tu', 'ns'], '4\n4\n')
    run([gmx, 'rmsf', '-s', 'md.tpr', '-f', 'md_fit.xtc', '-o', 'rmsf_ca.xvg', '-res', '-fit'], '3\n')
    run([gmx, 'gyrate', '-s', 'md.tpr', '-f', 'md_fit.xtc', '-o', 'gyrate.xvg'], '1\n')
    run([gmx, 'hbond', '-s', 'md.tpr', '-f', 'md_fit.xtc', '-num', 'hbond.xvg', '-o', 'hbond.ndx'], '1\n1\n')
    for stage in ('npt', 'md'):
        run([gmx, 'energy', '-f', f'{stage}.edr', '-o', f'{stage}_thermodynamics.xvg'], 'Temperature\nPressure\nDensity\nVolume\n\n')
    run([gmx, 'check', '-f', 'md.xtc'])
    save(out / 'complete.json', {'completed_unix': time.time(), 'cell': args.cell, 'seed': args.seed,
                                'protocol_sha256': identity['protocol_sha256'],
                                'trajectory_sha256': sha(out / 'md.xtc'), 'tpr_sha256': sha(out / 'md.tpr'),
                                'output_sha256': {f.name: sha(f) for f in out.iterdir() if f.is_file() and f.suffix in ('.xtc', '.tpr', '.xvg', '.cpt', '.gro', '.log')}})


if __name__ == '__main__':
    main()
