#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p .tools/bin
if [[ ! -x .tools/bin/micromamba ]]; then
  curl -fsSL https://micro.mamba.pm/api/micromamba/osx-arm64/latest -o .tools/micromamba.tar.bz2
  tar -xjf .tools/micromamba.tar.bz2 -C .tools bin/micromamba
fi
if [[ ! -x .conda/bin/python ]]; then
  if [[ -f config/environment.yml ]]; then
    .tools/bin/micromamba create -y -p "$ROOT/.conda" -f "$ROOT/config/environment.yml"
  else
    .tools/bin/micromamba create -y -p "$ROOT/.conda" -c conda-forge \
      python=3.11 vina meeko rdkit numpy matplotlib spyrmsd setuptools=80.9.0
  fi
fi
brew install gromacs pymol
.conda/bin/vina --version
gmx --version
pymol --version
