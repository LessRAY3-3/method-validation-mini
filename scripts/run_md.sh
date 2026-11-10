#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GMX="${GMX:-gmx}"
mkdir -p results/md logs
exec > >(tee logs/md_workflow.log) 2>&1

"$ROOT/.conda/bin/python" scripts/prepare_md_input.py
"$GMX" pdb2gmx -f data/prepared/1L2Y_model1.pdb -o results/md/prepared.gro \
  -p results/md/topol.top -i posre.itp \
  -ff amber99sb-ildn -water tip3p -ignh
mv posre.itp results/md/posre.itp
"$GMX" editconf -f results/md/prepared.gro -o results/md/boxed.gro \
  -c -d 1.0 -bt cubic
"$GMX" solvate -cp results/md/boxed.gro -cs spc216 \
  -o results/md/solvated.gro -p results/md/topol.top
"$GMX" grompp -f config/ions.mdp -c results/md/solvated.gro \
  -p results/md/topol.top -o results/md/ions.tpr -maxwarn 1
printf '13\n' | "$GMX" genion -s results/md/ions.tpr -o results/md/ionized.gro \
  -p results/md/topol.top -pname NA -nname CL -neutral
"$GMX" grompp -f config/minim.mdp -c results/md/ionized.gro \
  -p results/md/topol.top -o results/md/minim.tpr
"$GMX" mdrun -s results/md/minim.tpr -deffnm results/md/minim -nt 4
"$GMX" grompp -f config/nvt.mdp -c results/md/minim.gro -r results/md/minim.gro \
  -p results/md/topol.top -o results/md/nvt.tpr
"$GMX" mdrun -s results/md/nvt.tpr -deffnm results/md/nvt -nt 4
"$GMX" grompp -f config/npt.mdp -c results/md/nvt.gro -r results/md/nvt.gro \
  -t results/md/nvt.cpt -p results/md/topol.top -o results/md/npt.tpr
"$GMX" mdrun -s results/md/npt.tpr -deffnm results/md/npt -nt 4
"$GMX" grompp -f config/md.mdp -c results/md/npt.gro -t results/md/npt.cpt \
  -p results/md/topol.top -o results/md/md.tpr
"$GMX" mdrun -s results/md/md.tpr -deffnm results/md/md -nt 4
printf 'Temperature\n\n' | "$GMX" energy -f results/md/nvt.edr -o results/md/nvt_temperature.xvg
printf 'Temperature\nPressure\n\n' | "$GMX" energy -f results/md/npt.edr -o results/md/npt_temperature_pressure.xvg
printf 'Temperature\nPressure\n\n' | "$GMX" energy -f results/md/md.edr -o results/md/md_temperature_pressure.xvg

printf '1\n0\n' | "$GMX" trjconv -s results/md/md.tpr -f results/md/md.xtc \
  -o results/md/md_centered.xtc -pbc mol -center
printf '4\n0\n' | "$GMX" trjconv -s results/md/md.tpr -f results/md/md_centered.xtc \
  -o results/md/md_fit.xtc -fit rot+trans
printf '4\n4\n' | "$GMX" rms -s results/md/md.tpr -f results/md/md_fit.xtc \
  -o results/md/rmsd_backbone.xvg -tu ns
printf '3\n' | "$GMX" rmsf -s results/md/md.tpr -f results/md/md_fit.xtc \
  -o results/md/rmsf_ca.xvg -res
printf '1\n' | "$GMX" gyrate -s results/md/md.tpr -f results/md/md_fit.xtc \
  -o results/md/gyrate.xvg
printf '1\n1\n' | "$GMX" hbond -s results/md/md.tpr -f results/md/md_fit.xtc \
  -num results/md/hbond.xvg
"$GMX" check -f results/md/md.xtc
"$ROOT/.conda/bin/python" scripts/analyze_md.py
