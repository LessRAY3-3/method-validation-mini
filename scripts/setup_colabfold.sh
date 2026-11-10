#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .colabfold/bin/python ]]; then
  .tools/bin/micromamba create -y -p "$ROOT/.colabfold" \
    -c conda-forge -c bioconda python=3.13 mmseqs2=18.8cc5c pip
fi
.colabfold/bin/python -m pip install -r config/colabfold-requirements.txt
.colabfold/bin/python -c 'from importlib.metadata import version; import jax; print("ColabFold", version("colabfold"), "JAX", jax.__version__)'
