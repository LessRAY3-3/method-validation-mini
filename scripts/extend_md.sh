#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GMX="${GMX:-gmx}"
INPUT_BASE="${1:-md}"
OUTPUT_BASE="${2:-md_extended}"
EXTEND_PS="${3:-5000}"
mkdir -p results/md logs
exec > >(tee "logs/${OUTPUT_BASE}.log") 2>&1

"$GMX" convert-tpr -s "results/md/${INPUT_BASE}.tpr" -extend "$EXTEND_PS" \
  -o "results/md/${OUTPUT_BASE}.tpr"
"$GMX" mdrun -s "results/md/${OUTPUT_BASE}.tpr" -cpi "results/md/${INPUT_BASE}.cpt" \
  -deffnm "results/md/${OUTPUT_BASE}" -nt 6 -noappend
for ext in log edr gro xtc; do
  matches=("results/md/${OUTPUT_BASE}".part*."${ext}")
  if [[ -f "${matches[0]}" ]]; then
    mv "${matches[0]}" "results/md/${OUTPUT_BASE}.${ext}"
  fi
done
printf 'Temperature\nPressure\n\n' | "$GMX" energy -f "results/md/${OUTPUT_BASE}.edr" \
  -o "results/md/${OUTPUT_BASE}_temperature_pressure.xvg"
"$GMX" check -f "results/md/${OUTPUT_BASE}.xtc"
