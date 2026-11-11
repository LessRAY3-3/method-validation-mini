"""Read-only machine inventory before moving experiments; does not install or run jobs."""
import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def probe(argv):
    executable = shutil.which(argv[0])
    if executable is None:
        return {'available': False}
    try:
        p = subprocess.run([executable, *argv[1:]], text=True, capture_output=True, timeout=30)
        return {'available': True, 'executable': executable, 'returncode': p.returncode,
                'output': (p.stdout + p.stderr).strip()}
    except subprocess.TimeoutExpired:
        return {'available': True, 'executable': executable, 'error': 'timed out after 30 seconds'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gmx', default='gmx')
    parser.add_argument('--vina', default='vina')
    parser.add_argument('--prediction-python', help='Optional Python from the prediction environment')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {'system': platform.system(), 'machine': platform.machine(), 'release': platform.release(),
              'logical_cpus': os.cpu_count(), 'gromacs': probe([args.gmx, '--version']),
              'vina': probe([args.vina, '--version']),
              'gpu': probe(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
              'cuda_toolkit': probe(['nvcc', '--version']),
              'scope': 'Inventory only. Actual GPU execution, numerical correctness and throughput require a separate smoke run.'}
    if args.prediction_python:
        report['prediction_devices'] = probe([args.prediction_python, '-c', 'import jax; print(jax.__version__); print(jax.devices())'])
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
