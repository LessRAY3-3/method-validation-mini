#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GMX="${GMX:-gmx}"
TOTAL_NS="${TOTAL_NS:-6}"
TAG="${TAG:-}"
if [[ "$TOTAL_NS" == 6 ]]; then
  LATEST="md_extended"
  PARTS=(results/md/md_fit.xtc results/md/md_extended_fit.xtc)
elif [[ "$TOTAL_NS" == 11 ]]; then
  LATEST="md_extended2"
  PARTS=(results/md/md_fit.xtc results/md/md_extended_fit.xtc results/md/md_extended2_fit.xtc)
else
  printf 'Supported TOTAL_NS values: 6, 11\n' >&2
  exit 2
fi
exec > >(tee "logs/md_stability${TAG}.log") 2>&1

if [[ ! -f "results/md/${LATEST}_fit.xtc" ]]; then
  printf '1\n0\n' | "$GMX" trjconv -s "results/md/${LATEST}.tpr" -f "results/md/${LATEST}.xtc" \
    -o "results/md/${LATEST}_centered.xtc" -pbc mol -center
  printf '4\n0\n' | "$GMX" trjconv -s "results/md/${LATEST}.tpr" -f "results/md/${LATEST}_centered.xtc" \
    -o "results/md/${LATEST}_fit.xtc" -fit rot+trans
fi
"$GMX" trjcat -f "${PARTS[@]}" -o "results/md/md_all_fit${TAG}.xtc"
"$GMX" check -f "results/md/md_all_fit${TAG}.xtc"
printf '4\n4\n' | "$GMX" rms -s results/md/md.tpr -f "results/md/md_all_fit${TAG}.xtc" \
  -o "results/md/rmsd_all${TAG}.xvg" -tu ns
printf '3\n' | "$GMX" rmsf -s results/md/md.tpr -f "results/md/md_all_fit${TAG}.xtc" \
  -o "results/md/rmsf_all${TAG}.xvg" -res
printf '1\n' | "$GMX" gyrate -s results/md/md.tpr -f "results/md/md_all_fit${TAG}.xtc" \
  -o "results/md/gyrate_all${TAG}.xvg"
printf '1\n1\n' | "$GMX" hbond -s results/md/md.tpr -f "results/md/md_all_fit${TAG}.xtc" \
  -num "results/md/hbond_all${TAG}.xvg"
for ((window=0; window<TOTAL_NS; window++)); do
  start=$((window * 1000))
  end=$((start + 1000))
  printf '3\n' | "$GMX" rmsf -s results/md/md.tpr -f "results/md/md_all_fit${TAG}.xtc" \
    -o "results/md/rmsf_ca_${start}_${end}${TAG}.xvg" -res -b "$start" -e "$end"
done
TOTAL_NS="$TOTAL_NS" TAG="$TAG" "$ROOT/.conda/bin/python" scripts/analyze_md_stability.py
