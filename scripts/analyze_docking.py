import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from spyrmsd import rmsd as spyrmsd_rmsd

root = Path(__file__).resolve().parents[1]
out_dir = root / "results/docking"
ref_path = root / "data/raw/1iep_ligand.sdf"
ref_input = Chem.SDMolSupplier(str(ref_path), removeHs=False)[0]
if ref_input is None:
    raise RuntimeError("Could not read the crystal ligand")
ref = Chem.RemoveHs(ref_input)
ref_coords = ref.GetConformer().GetPositions()
ref_atomicnums = np.array([atom.GetAtomicNum() for atom in ref.GetAtoms()])

rows = []
for sdf_path in sorted(out_dir.glob("vina_e*_s*.sdf")):
    match = re.fullmatch(r"vina_e(\d+)_s(\d+)", sdf_path.stem)
    if not match:
        continue
    exhaustiveness, seed = map(int, match.groups())
    pdbqt_path = sdf_path.with_suffix(".pdbqt")
    scores = []
    for line in pdbqt_path.read_text().splitlines():
        score_match = re.match(r"REMARK VINA RESULT:\s+(-?[\d.]+)", line)
        if score_match:
            scores.append(float(score_match.group(1)))
    poses = list(Chem.SDMolSupplier(str(sdf_path), removeHs=False))
    if len(poses) != len(scores):
        raise RuntimeError(f"Pose/score count mismatch in {sdf_path}")
    for mode, mol in enumerate(poses, start=1):
        if mol is None:
            raise RuntimeError(f"Unreadable pose {mode} in {sdf_path}")
        mol = Chem.RemoveHs(mol)
        coords = mol.GetConformer().GetPositions()
        atomicnums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        atom_map = ref.GetSubstructMatch(mol)
        if not atom_map:
            raise RuntimeError(f"No reference-to-pose atom map for mode {mode} in {sdf_path}")
        atom_map_array = np.array(atom_map)
        mapped_ref_coords = ref_coords[atom_map_array]
        mapped_ref_atomicnums = ref_atomicnums[atom_map_array]
        plain_value = spyrmsd_rmsd.rmsd(
            mapped_ref_coords, coords, mapped_ref_atomicnums, atomicnums, center=False
        )
        ref_adjacency = Chem.GetAdjacencyMatrix(ref)[np.ix_(atom_map_array, atom_map_array)]
        pose_adjacency = Chem.GetAdjacencyMatrix(mol)
        value = spyrmsd_rmsd.symmrmsd(
            mapped_ref_coords,
            coords,
            mapped_ref_atomicnums,
            atomicnums,
            ref_adjacency,
            pose_adjacency,
            center=False,
            minimize=False,
        )
        rows.append({
            "exhaustiveness": exhaustiveness,
            "seed": seed,
            "mode": mode,
            "affinity_kcal_mol": scores[mode - 1],
            "crystal_heavy_atom_rmsd_angstrom": float(value),
            "crystal_heavy_atom_plain_rmsd_angstrom": float(plain_value),
        })

if not rows:
    raise RuntimeError("No docking poses found")
rows.sort(key=lambda row: (row["exhaustiveness"], row["seed"], row["mode"]))
fieldnames = list(rows[0])
with (out_dir / "all_poses.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

summaries = []
for key in sorted({(row["exhaustiveness"], row["seed"]) for row in rows}):
    run_rows = [row for row in rows if (row["exhaustiveness"], row["seed"]) == key]
    top = min(run_rows, key=lambda row: row["affinity_kcal_mol"])
    closest = min(run_rows, key=lambda row: row["crystal_heavy_atom_rmsd_angstrom"])
    summaries.append({
        "exhaustiveness": key[0],
        "seed": key[1],
        "top_score_mode": top["mode"],
        "top_score_affinity_kcal_mol": top["affinity_kcal_mol"],
        "top_score_rmsd_angstrom": top["crystal_heavy_atom_rmsd_angstrom"],
        "minimum_rmsd_mode": closest["mode"],
        "minimum_rmsd_angstrom": closest["crystal_heavy_atom_rmsd_angstrom"],
        "minimum_rmsd_affinity_kcal_mol": closest["affinity_kcal_mol"],
    })
with (out_dir / "run_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)

ligand_centroid = ref_coords.mean(axis=0)
metadata = {
    "reference": "1IEP STI crystal ligand, hydrogens removed",
    "rmsd_method": "spyrmsd symmetry-corrected heavy-atom RMSD; no centering or alignment",
    "reference_heavy_atoms": len(ref_atomicnums),
    "crystal_ligand_centroid_angstrom": ligand_centroid.tolist(),
    "common_redocking_reference_cutoff_angstrom": 2.0,
    "runs": summaries,
}
(out_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

fig, ax = plt.subplots(figsize=(7.2, 4.8))
for key in sorted({(row["exhaustiveness"], row["seed"]) for row in rows}):
    run_rows = [row for row in rows if (row["exhaustiveness"], row["seed"]) == key]
    ax.scatter(
        [row["crystal_heavy_atom_rmsd_angstrom"] for row in run_rows],
        [row["affinity_kcal_mol"] for row in run_rows],
        label=f"e={key[0]}, seed={key[1]}", s=42, alpha=0.8,
    )
ax.axvline(2.0, color="black", linestyle="--", linewidth=1, label="2 Å reference")
ax.set(xlabel="Crystal ligand heavy-atom RMSD (Å)", ylabel="Vina score (kcal/mol)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(root / "figures/docking_score_vs_rmsd.png", dpi=200)
print(json.dumps(metadata, indent=2))
