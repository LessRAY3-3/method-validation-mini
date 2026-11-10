import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
pred_dir = root / "results/prediction"
fig_dir = root / "figures"


def ca_plddt(path):
    residues = []
    scores = []
    seen = set()
    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        residue = (line[21].strip(), int(line[22:26]), line[26].strip())
        if residue in seen:
            continue
        seen.add(residue)
        residues.append(residue[1])
        scores.append(float(line[60:66]))
    return np.array(residues), np.array(scores)


def plot_plddt(residues, scores, output):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(residues, scores, marker="o")
    ax.axhspan(90, 100, color="navy", alpha=0.08)
    ax.axhspan(70, 90, color="skyblue", alpha=0.10)
    ax.axhspan(50, 70, color="gold", alpha=0.10)
    ax.axhspan(0, 50, color="orange", alpha=0.08)
    ax.set(xlabel="Residue number", ylabel="pLDDT", ylim=(0, 100), xticks=residues)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / output, dpi=200)
    plt.close(fig)


summary = {}
esmfold = pred_dir / "esmfold_1L2Y.pdb"
if esmfold.exists():
    residues, scores = ca_plddt(esmfold)
    input_scale = "0–1" if scores.max() <= 1.0 else "0–100"
    if input_scale == "0–1":
        scores = scores * 100.0
    plot_plddt(residues, scores, "prediction_esmfold_plddt.png")
    summary["ESMFold"] = {
        "source": "ESM Atlas API foldSequence/v1/pdb",
        "run_date": "2026-11-10",
        "sequence_length": len(residues),
        "plddt_from": "PDB B-factor field at Cα atoms",
        "input_plddt_scale": input_scale,
        "mean_plddt": float(scores.mean()),
        "minimum_plddt": float(scores.min()),
        "minimum_plddt_residue": int(residues[scores.argmin()]),
        "pae": "not provided by this endpoint",
    }

for pae_path in sorted(pred_dir.rglob("*scores*.json")):
    data = json.loads(pae_path.read_text())
    if "plddt" not in data or "pae" not in data:
        continue
    plddt = np.asarray(data["plddt"], dtype=float)
    pae = np.asarray(data["pae"], dtype=float)
    residues = np.arange(1, len(plddt) + 1)
    plot_plddt(residues, plddt, "prediction_colabfold_plddt.png")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    image = ax.imshow(pae, cmap="viridis", origin="lower", vmin=0)
    ax.set(xlabel="Residue number", ylabel="Residue number")
    fig.colorbar(image, ax=ax, label="PAE (Å)")
    fig.tight_layout()
    fig.savefig(fig_dir / "prediction_colabfold_pae.png", dpi=200)
    plt.close(fig)
    summary["ColabFold"] = {
        "scores_file": pae_path.name,
        "mean_plddt": float(plddt.mean()),
        "mean_pae_angstrom": float(pae.mean()),
        "maximum_pae_angstrom": float(pae.max()),
    }

if not summary:
    raise RuntimeError("No prediction outputs found")
(pred_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
