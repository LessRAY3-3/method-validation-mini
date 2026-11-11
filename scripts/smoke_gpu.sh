#!/usr/bin/env bash
# Isolated 20 ps NVT smoke test; never counted as a formal replicate.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GMX="${GMX:-$ROOT/env/bin/gmx}"
WORK="$ROOT/setup/gpu_smoke"
mkdir -p "$WORK"
"$GMX" --version > "$ROOT/setup/gromacs_version.txt" 2>&1
"$GMX" dump -s "$ROOT/results/md/md.tpr" > /dev/null 2> "$ROOT/setup/original_tpr_read.log"
cp "$ROOT/results/md/topol.top" "$ROOT/results/md/posre.itp" "$WORK/"
sed 's/^nsteps = .*/nsteps = 10000/' "$ROOT/config/nvt.mdp" > "$WORK/smoke.mdp"
cd "$WORK"
"$GMX" grompp -f smoke.mdp -c "$ROOT/results/md/minim.gro" -r "$ROOT/results/md/minim.gro" -p topol.top -o smoke.tpr -po smoke_processed.mdp
"$GMX" mdrun -s smoke.tpr -deffnm smoke -ntmpi 1 -ntomp 4 -nb gpu -pme auto -gpu_id 0
# A smoke test explicitly requires GPU use; failure must not silently fall back.
grep -E 'GPU|Performance:' smoke.log
