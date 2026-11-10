#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/.conda/bin"
cd "$ROOT"
mkdir -p results/docking logs

run_one() {
  local exhaustiveness="$1"
  local seed="$2"
  local stem="results/docking/vina_e${exhaustiveness}_s${seed}"
  "$BIN/vina" \
    --receptor data/prepared/1iep_receptor.pdbqt \
    --ligand data/prepared/1iep_ligand.pdbqt \
    --config config/1iep_box.txt \
    --scoring vina \
    --exhaustiveness "$exhaustiveness" \
    --num_modes 9 \
    --energy_range 3 \
    --seed "$seed" \
    --cpu 4 \
    --out "${stem}.pdbqt" \
    2>&1 | tee "logs/docking_e${exhaustiveness}_s${seed}.log"
  "$BIN/python" "$BIN/mk_export.py" "${stem}.pdbqt" -s "${stem}.sdf"
}

run_one 8 42
run_one 32 42
run_one 32 2026
run_one 32 20261110

"$BIN/python" scripts/analyze_docking.py
