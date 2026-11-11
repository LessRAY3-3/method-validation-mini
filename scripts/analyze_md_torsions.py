"""Post-hoc torsion diagnosis of saved MD; no dynamics, no significance tests."""
import csv,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.vectors import Vector,calc_dihedral
from analyze_md_replicates import read_gro_trajectory,validate_times
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/md_deep_dive/torsions'
SEEDS=(20261110,20261111,20261112)
WINDOWS=((0,1),(1,3),(3,8),(8,11),(0,11))

def delta(a,b,period=360):return (np.asarray(a)-b+period/2)%period-period/2

def torsion(x):
    b0=x[...,0,:]-x[...,1,:]; b1=x[...,2,:]-x[...,1,:]; b2=x[...,3,:]-x[...,2,:]
    b1=b1/np.linalg.norm(b1,axis=-1,keepdims=True)
    v=b0-np.sum(b0*b1,axis=-1,keepdims=True)*b1
    w=b2-np.sum(b2*b1,axis=-1,keepdims=True)*b1
    if np.any(np.linalg.norm(v,axis=-1)<1e-10) or np.any(np.linalg.norm(w,axis=-1)<1e-10):raise ValueError('Undefined torsion')
    return np.degrees(np.arctan2(np.sum(np.cross(b1,v)*w,axis=-1),np.sum(v*w,axis=-1)))

def circular_summary(values,period=360):
    z=np.exp(2j*np.pi*np.asarray(values)/period).mean()
    return float(np.angle(z)*period/(2*np.pi)),float(abs(z))

def definitions():
    d={}
    for i in range(1,21):
        if i>1:d[f'{i}_phi']=([(i-1,'C'),(i,'N'),(i,'CA'),(i,'C')],360)
        if i<20:d[f'{i}_psi']=([(i,'N'),(i,'CA'),(i,'C'),(i+1,'N')],360)
    for i in (3,6):
        d[f'{i}_chi1']=([(i,'N'),(i,'CA'),(i,'CB'),(i,'CG')],360)
        d[f'{i}_chi2']=([(i,'CA'),(i,'CB'),(i,'CG'),(i,'CD1')],180 if i==3 else 360)
    return d

def writecsv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():raise ValueError('Existing completed output; preserve it')
    protocol={'declared_utc':datetime.now(timezone.utc).isoformat(),'design':'Exploratory post-hoc; seed20261110 early deviation was already observed. All seeds, residues and declared windows retained.','seeds':SEEDS,'windows_ns':WINDOWS,'snapshots_ns':[1.5,3.5],'snapshot_selection':'Fixed illustrative times chosen after seeing early deviation, not independent validation or peak selection','torsions':{k:{'atoms':v[0],'period_deg':v[1]} for k,v in definitions().items()},'limitations':'38 NMR models are reference diversity, not independent sampling or confidence intervals. No frame-level significance test or transition-rate inference.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    models=list(PDBParser(QUIET=True).get_structure('nmr',ROOT/'data/raw/1L2Y.pdb').get_models());assert len(models)==38
    refs=[]
    for m in models:
        refs.append({(r.id[1],a.name):np.asarray(a.coord,dtype=float) for r in m.get_residues() for a in r})
    ds=definitions();refangles={k:torsion(np.array([[ref[a] for a in atoms] for ref in refs])) for k,(atoms,p) in ds.items()}
    writecsv(OUT/'nmr_torsions.csv',[{'nmr_model':j+1,'torsion':k,'degrees':float(a),'period_deg':ds[k][1]} for k,arr in refangles.items() for j,a in enumerate(arr)])
    rows=[];checks=[];hashes={};series={}
    for seed in SEEDS:
        source=ROOT/f'results/md_replicates_gpu/aggregate_analysis_audited/seed_{seed}/protein_whole.gro'
        hashes[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
        time,xyz,ids=read_gro_trajectory(source);validate_times(time,11)
        idx={(r,n):i for i,(r,_,n) in enumerate(ids)}
        angles={k:torsion(xyz[:,[idx[a] for a in atoms]]) for k,(atoms,_) in ds.items()}
        series[seed]=angles
        writecsv(OUT/f'seed_{seed}_torsions.csv',[{'time_ps':float(t),**{k:float(v[i]) for k,v in angles.items()}} for i,t in enumerate(time)])
        for key,(atoms,period) in ds.items():
            for frame in (0,150,350,800,1100):
                bio=np.degrees(calc_dihedral(*[Vector(x) for x in xyz[frame,[idx[a] for a in atoms]]]))
                err=abs(float(delta(angles[key][frame],bio)))
                assert err<1e-8
                checks.append(err)
            nearest=np.min(abs(delta(angles[key][:,None],refangles[key][None,:],period)),axis=1)
            for a,b in WINDOWS:
                mask=(time>=a*1000)&((time<=b*1000) if b==11 else (time<b*1000))
                mean,resultant=circular_summary(angles[key][mask],period)
                rows.append({'seed':seed,'window_ns':f'{a}-{b}','torsion':key,'frames':int(mask.sum()),'period_deg':period,'circular_mean_deg':mean,'circular_resultant_length':resultant,'mean_nearest_nmr_angular_difference_deg':float(nearest[mask].mean()),'median_nearest_nmr_angular_difference_deg':float(np.median(nearest[mask]))})
        ca=[idx[i,'CA'] for i in range(1,21)];refca=np.array([refs[0][i,'CA'] for i in range(1,21)])
        for t in (1500,3500):
            frame=int(t/10);x=xyz[frame];mobile=x[ca];u,_,vt=np.linalg.svd((mobile-mobile.mean(0)).T@(refca-refca.mean(0)));cor=np.eye(3);cor[-1,-1]=np.linalg.det(u@vt);rot=u@cor@vt;aligned=(x-mobile.mean(0))@rot+refca.mean(0)
            lines=[]
            for j,((res,resname,name),coord) in enumerate(zip(ids,aligned),1):
                element=name.lstrip('0123456789')[0]
                atomfield=f' {name:<3}' if len(name)<4 else name
                lines.append(f'ATOM  {j:5d} {atomfield} {resname:3} A{res:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00          {element:>2}')
            (OUT/f'seed_{seed}_{t}ps.pdb').write_text('\n'.join(lines)+'\nEND\n')
    writecsv(OUT/'window_torsion_summary.csv',rows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for ax,key in zip(axs,['3_chi1','3_chi2','19_phi','19_psi']):
        period=ds[key][1]
        for seed in SEEDS:ax.scatter(time/1000,delta(series[seed][key],0,period),s=2,label=str(seed),alpha=.45)
        for angle in refangles[key]:ax.axhline(float(delta(angle,0,period)),color='gray',lw=.4,alpha=.18)
        ax.axvspan(1,3,color='orange',alpha=.08);ax.set_ylabel(key+' (degrees)');ax.set_ylim(-period/2,period/2)
    axs[0].legend(markerscale=3);axs[-1].set_xlabel('Production time (ns); shaded 1-3 ns is post-hoc diagnostic window')
    fig.suptitle('All three trajectories; gray lines: 38 NMR reference angles; Tyr chi2 modulo 180°')
    fig.tight_layout();fig.savefig(OUT/'torsion_timeseries.png',dpi=180);plt.close(fig)
    pml=['reinitialize','bg_color white','set ray_opaque_background, 1','set cartoon_fancy_helices, 1']
    for seed,t in [(s,t) for s in SEEDS for t in (1500,3500)]:
        name=f's{seed}_{t}ps'
        pml.extend([f'load {OUT/f"seed_{seed}_{t}ps.pdb"}, {name}',f'color cyan, {name}'])
    pml.extend([f'load {ROOT/"data/prepared/1L2Y_model1.pdb"}, reference','color gray70, reference','hide everything','show cartoon','show sticks, resi 3+19 and not elem H','set stick_radius, 0.13','orient reference','zoom all, 2'])
    for seed,t in [(s,t) for s in SEEDS for t in (1500,3500)]:
        pml.extend(['disable all','enable reference',f'enable s{seed}_{t}ps',f'png {OUT/f"snapshot_{seed}_{t}.png"}, 600, 450, ray=1'])
    pml.append('quit')
    (OUT/'fixed_snapshots.pml').write_text('\n'.join(pml)+'\n')
    (OUT/'summary.json').write_text(json.dumps({'status':'complete','source_hashes':hashes,'independent_biopython_checks':len(checks),'maximum_circular_error_deg':max(checks),'windows':WINDOWS,'atom_definitions':protocol['torsions'],'note':'Circular resultant length describes concentration, not confidence. Tyr chi2 period180 accounts for aromatic label symmetry. Closest NMR angular difference is a descriptive minimum across 38 references, not population likelihood.'},indent=2)+'\n')
    print('completed',OUT,'checks',len(checks))

if __name__=='__main__':main()
