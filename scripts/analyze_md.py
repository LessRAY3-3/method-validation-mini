import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
md_dir = root / "results/md"
figure_dir = root / "figures"


def load_xvg(name):
    return np.loadtxt(md_dir / name, comments=["#", "@"])


def time_plot(name, ylabel, output, ycolumn=1):
    data = load_xvg(name)
    time = data[:, 0]
    values = data[:, ycolumn]
    if name.startswith(("gyrate", "hbond")):
        time = time / 1000.0
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(time, values, linewidth=1.2)
    ax.set(xlabel="Time (ns)", ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(figure_dir / output, dpi=200)
    plt.close(fig)
    return {
        "frames": int(len(values)),
        "start": float(values[0]),
        "end": float(values[-1]),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


summary = {
    "rmsd_backbone_nm": time_plot(
        "rmsd_backbone.xvg", "Backbone RMSD (nm)", "md_rmsd.png"
    ),
    "radius_of_gyration_nm": time_plot(
        "gyrate.xvg", "Radius of gyration (nm)", "md_gyrate.png"
    ),
    "protein_internal_hbonds": time_plot(
        "hbond.xvg", "Protein–protein hydrogen bonds", "md_hbond.png"
    ),
}
rmsf = load_xvg("rmsf_ca.xvg")
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(rmsf[:, 0], rmsf[:, 1], marker="o", markersize=3)
ax.set(xlabel="Residue number", ylabel="Cα RMSF (nm)", xticks=rmsf[:, 0].astype(int))
ax.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(figure_dir / "md_rmsf.png", dpi=200)
plt.close(fig)
summary["rmsf_ca_nm"] = {
    "residues": int(len(rmsf)),
    "mean": float(np.mean(rmsf[:, 1])),
    "maximum": float(np.max(rmsf[:, 1])),
    "maximum_residue": int(rmsf[np.argmax(rmsf[:, 1]), 0]),
}
summary["limitations"] = "1 ns single trajectory; not evidence of convergence, folding, thermodynamic stability, or binding mechanism."
(md_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
