#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GMX="${GMX:-gmx}"
TOTAL_NS="${TOTAL_NS:-6}"
TAG="${TAG:-}"
if [[ "$TOTAL_NS" == 6 ]]; then
  BASES=(md md_extended)
  PARTS=(results/md/md_fit.xtc results/md/md_extended_fit.xtc)
elif [[ "$TOTAL_NS" == 11 ]]; then
  BASES=(md md_extended md_extended2)
  PARTS=(results/md/md_fit.xtc results/md/md_extended_fit.xtc results/md/md_extended2_fit.xtc)
else
  printf 'Supported TOTAL_NS values: 6, 11\n' >&2
  exit 2
fi
mkdir -p logs
# Direct redirection also works in sandboxes that restrict /dev/fd.
printf 'Analysis progress: logs/md_stability%s.log\n' "$TAG"
exec > "logs/md_stability${TAG}.log" 2>&1

# Always regenerate every segment: an existing fit may belong to an older run.
for LATEST in "${BASES[@]}"; do
  printf '1\n0\n' | "$GMX" trjconv -s "results/md/${LATEST}.tpr" -f "results/md/${LATEST}.xtc" \
    -o "results/md/${LATEST}_centered.xtc" -pbc mol -center
  printf '4\n0\n' | "$GMX" trjconv -s "results/md/${LATEST}.tpr" -f "results/md/${LATEST}_centered.xtc" \
    -o "results/md/${LATEST}_fit.xtc" -fit rot+trans
done
"$GMX" trjcat -f "${PARTS[@]}" -o "results/md/md_all_fit${TAG}.xtc"
"$GMX" check -f "results/md/md_all_fit${TAG}.xtc"
printf '4\n4\n' | "$GMX" rms -s results/md/md.tpr -f "results/md/md_all_fit${TAG}.xtc" \
  -o "results/md/rmsd_all${TAG}.xvg" -tu ns
printf '3\n' | "$GMX" rmsf -s results/md/md.tpr -f "results/md/md_all_fit${TAG}.xtc" \
  -o "results/md/rmsf_all${TAG}.xvg" -res -fit
printf '1\n' | "$GMX" gyrate -s results/md/md.tpr -f "results/md/md_all_fit${TAG}.xtc" \
  -o "results/md/gyrate_all${TAG}.xvg"
printf '1\n1\n' | "$GMX" hbond -s results/md/md.tpr -f "results/md/md_all_fit${TAG}.xtc" \
  -num "results/md/hbond_all${TAG}.xvg"
for ((window=0; window<TOTAL_NS; window++)); do
  start=$((window * 1000))
  end=$((start + 1000))
  # GROMACS -e is inclusive; omit the next window's first frame.
  analysis_end="$end"
  if (( window < TOTAL_NS - 1 )); then analysis_end=$((end - 10)); fi
  printf '3\n' | "$GMX" rmsf -s results/md/md.tpr -f "results/md/md_all_fit${TAG}.xtc" \
    -o "results/md/rmsf_ca_${start}_${end}${TAG}.xvg" -res -fit -b "$start" -e "$analysis_end"
done
TOTAL_NS="$TOTAL_NS" TAG="$TAG" "$ROOT/.conda/bin/python" scripts/analyze_md_stability.py
