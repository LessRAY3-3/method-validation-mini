#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/.conda/bin"
cd "$ROOT"
mkdir -p data/prepared logs

"$BIN/python" "$BIN/mk_prepare_receptor.py" \
  -i data/raw/1iep_receptorH.pdb \
  -o data/prepared/1iep_receptor \
  -p -v \
  --box_size 20 20 20 \
  --box_center 15.190 53.903 16.917 \
  2>&1 | tee logs/prepare_receptor.log

"$BIN/python" "$BIN/mk_prepare_ligand.py" \
  -i data/raw/1iep_ligand.sdf \
  -o data/prepared/1iep_ligand.pdbqt \
  2>&1 | tee logs/prepare_ligand.log
