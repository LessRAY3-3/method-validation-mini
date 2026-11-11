"""Render the fixed post-hoc MD illustrations from already exported PDB files."""
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/md_deep_dive/torsions'
subprocess.run(['pymol','-cq',str(OUT/'fixed_snapshots.pml')],check=True,cwd=ROOT)
fig,axs=plt.subplots(3,2,figsize=(11,12))
for row,seed in enumerate((20261110,20261111,20261112)):
 for col,t in enumerate((1500,3500)):
  ax=axs[row,col];ax.imshow(plt.imread(OUT/f'snapshot_{seed}_{t}.png'));ax.set_title(f'Seed {seed} | {t/1000:.1f} ns');ax.axis('off')
fig.suptitle('Fixed illustrative times, chosen post hoc\nCyan: trajectory; gray: NMR model 1; sticks: Y3/P19; common all-CA fit',fontsize=12)
fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'fixed_snapshots.png',dpi=160);plt.close(fig)
