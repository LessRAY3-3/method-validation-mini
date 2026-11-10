import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
md_dir = root / "results/md"
fig_dir = root / "figures"
total_ns = int(os.environ.get("TOTAL_NS", "6"))
tag = os.environ.get("TAG", "")


def load(name):
    return np.loadtxt(md_dir / name, comments=["#", "@"])


def window_values(data, index, time_scale):
    time = data[:, 0] * time_scale
    values = data[:, index]
    result = []
    for start in range(total_ns):
        selected = (time >= start) & (time <= start + 1)
        x = time[selected]
        y = values[selected]
        slope = np.polyfit(x, y, 1)[0]
        result.append((float(y.mean()), float(y.std()), float(slope)))
    return result


rmsd = load(f"rmsd_all{tag}.xvg")
gyrate = load(f"gyrate_all{tag}.xvg")
hbond = load(f"hbond_all{tag}.xvg")
window_metrics = {
    "rmsd": window_values(rmsd, 1, 1.0),
    "gyrate": window_values(gyrate, 1, 0.001),
    "hbond": window_values(hbond, 1, 0.001),
}
rows = []
for window in range(total_ns):
    rmsf = load(f"rmsf_ca_{window * 1000}_{(window + 1) * 1000}{tag}.xvg")
    row = {
        "window_ns": f"{window}–{window + 1}",
        "rmsd_mean_nm": window_metrics["rmsd"][window][0],
        "rmsd_within_window_sd_nm": window_metrics["rmsd"][window][1],
        "rmsd_slope_nm_per_ns": window_metrics["rmsd"][window][2],
        "gyrate_mean_nm": window_metrics["gyrate"][window][0],
        "hbond_mean": window_metrics["hbond"][window][0],
        "rmsf_mean_nm": float(rmsf[:, 1].mean()),
        "rmsf_max_nm": float(rmsf[:, 1].max()),
        "rmsf_max_residue": int(rmsf[np.argmax(rmsf[:, 1]), 0]),
    }
    rows.append(row)
with (md_dir / f"stability_windows{tag}.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

last = rows[-3:]
criteria = {
    "rmsd_last_3_window_mean_range_nm_max_0.03": float(np.ptp([r["rmsd_mean_nm"] for r in last])) <= 0.03,
    "rmsd_last_3_window_absolute_slopes_nm_per_ns_max_0.03": bool(all(abs(r["rmsd_slope_nm_per_ns"]) <= 0.03 for r in last)),
    "gyrate_last_3_window_mean_range_nm_max_0.02": float(np.ptp([r["gyrate_mean_nm"] for r in last])) <= 0.02,
    "hbond_last_3_window_mean_range_max_2": float(np.ptp([r["hbond_mean"] for r in last])) <= 2.0,
}
summary = {
    "trajectory": f"continuous {total_ns} ns trajectory analyzed in {total_ns} one-ns windows",
    "frames_expected": total_ns * 100 + 1,
    "criteria": criteria,
    "all_project_plateau_criteria_met": bool(all(criteria.values())),
    "interpretation": "These are project-specific descriptive heuristics, not proof of thermodynamic convergence or folding stability.",
    "temperature_pressure_files": ["md_temperature_pressure.xvg", "md_extended_temperature_pressure.xvg"] + (["md_extended2_temperature_pressure.xvg"] if total_ns == 11 else []),
}
(md_dir / f"stability_summary{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")

fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
x = np.arange(1, total_ns + 1)
axes[0].errorbar(x, [r["rmsd_mean_nm"] for r in rows], yerr=[r["rmsd_within_window_sd_nm"] for r in rows], marker="o", capsize=3)
axes[0].set_ylabel("Backbone RMSD (nm)")
axes[1].plot(x, [r["gyrate_mean_nm"] for r in rows], marker="o")
axes[1].set_ylabel("Mean Rg (nm)")
axes[2].plot(x, [r["hbond_mean"] for r in rows], marker="o")
axes[2].set_ylabel("Mean internal H-bonds")
axes[2].set_xlabel("1 ns window")
axes[2].set_xticks(x, [f"{i}–{i+1}" for i in range(total_ns)])
fig.tight_layout()
fig.savefig(fig_dir / f"md_stability_windows{tag}.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.5))
for window in range(total_ns):
    rmsf = load(f"rmsf_ca_{window * 1000}_{(window + 1) * 1000}{tag}.xvg")
    ax.plot(rmsf[:, 0], rmsf[:, 1], label=f"{window}–{window + 1} ns", alpha=0.8)
ax.set(xlabel="Residue number", ylabel="Cα RMSF (nm)", xticks=range(1, 21))
ax.tick_params(axis="x", labelsize=7)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(fig_dir / f"md_rmsf_windows{tag}.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.0, 4.2))
ax.plot(rmsd[:, 0], rmsd[:, 1], linewidth=1.0)
ax.set(xlabel="Time (ns)", ylabel="Backbone RMSD (nm)")
fig.tight_layout()
fig.savefig(fig_dir / f"md_extended_rmsd{tag}.png", dpi=200)
plt.close(fig)
print(json.dumps(summary, indent=2))
print(json.dumps(rows, indent=2))
