"""Read-only independent checks of completed MD outputs; no simulation."""
import argparse, csv, hashlib, json, re, subprocess
from pathlib import Path
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from analyze_md_replicates import read_gro_trajectory
from prediction_utils import ca_models, fasta_sequence
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gmx', default='gmx')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/md_replicates_gpu'
OUT=BASE/'aggregate_analysis_audited'
ref=ca_models(ROOT/'data/raw/1L2Y.pdb',fasta_sequence(ROOT/'data/prepared/1L2Y.fasta'))[0][1]
results=[]
for seed in (20261110,20261111,20261112):
 d=BASE/f'seed_{seed}';a=OUT/f'seed_{seed}'
 dump=subprocess.run([args.gmx,'dump','-s',str(d/'nvt.tpr')],capture_output=True,text=True,check=True).stdout
 hashes={name:hashlib.sha256('\n'.join(re.findall(r'^   '+name+r'\[.*$',dump,re.M)).encode()).hexdigest() for name in ('x','v')}
 assert len(re.findall(r'^   v\[',dump,re.M))>8000
 stage=[]
 for label in ('nvt','npt','md'):
  text=(d/f'{label}_processed.mdp').read_text();log=(d/f'{label}.log').read_text()
  params=dict(re.findall(r'^([\w_-]+)\s*=\s*([^\n;]+)',text,re.M))
  stage.append({'stage':label,'ld_seed':params.get('ld-seed','').strip(),'gen_vel':params.get('gen_vel','').strip(),'gen_seed':params.get('gen_seed','').strip(),'lincs_warning_count':log.count('LINCS WARNING'),'fatal_error_count':log.count('Fatal error'),'gpu_short_range':'short-ranged interactions on the GPU' in log,'gpu_pme':'PME tasks will do all aspects on the GPU' in log})
  assert int(params['ld-seed'])==seed
 energy=np.loadtxt(d/'temperature_pressure.xvg',comments=['@','#'])
 assert energy.shape==(11001,3) and np.isfinite(energy).all()
 ts,xyz,ids=read_gro_trajectory(a/'protein_whole.gro')
 series=list(csv.DictReader((a/'timeseries.csv').open()))
 old=list(csv.DictReader((BASE/'aggregate_analysis'/f'seed_{seed}'/'timeseries.csv').open()))
 ca=[i for i,(_,_,name) in enumerate(ids) if name=='CA']
 checks=[]
 for idx in (0,800,1100):
  sup=SVDSuperimposer();sup.set(ref,xyz[idx,ca]);sup.run()
  q=[]
  for i in range(20):
   for j in range(i+3,20):
    distance=np.linalg.norm(ref[i]-ref[j])
    if distance<=8:q.append(np.linalg.norm(xyz[idx,ca[i]]-xyz[idx,ca[j]])<=1.2*distance)
  row=series[idx]
  check={'frame':idx,'rmsd_error_angstrom':abs(sup.get_rms()-float(row['ca_rmsd_nmr1_angstrom'])),'q_error':abs(np.mean(q)-float(row['q_ca_8A']))}
  assert check['rmsd_error_angstrom']<1e-10 and check['q_error']<1e-10
  checks.append(check)
 deltas={key:max(abs(float(x[key])-float(y[key])) for x,y in zip(series,old)) for key in old[0] if key!='time_ps'}
 results.append({'seed':seed,'nvt_initial_arrays_sha256':hashes,'stage_health':stage,'energy_samples':len(energy),'temperature_mean_K':float(energy[:,1].mean()),'temperature_temporal_sd_K':float(energy[:,1].std()),'pressure_mean_bar':float(energy[:,2].mean()),'pressure_temporal_sd_bar':float(energy[:,2].std()),'independent_sample_checks':checks,'original_vs_local_reanalysis_max_absolute_difference':deltas})
assert len({r['nvt_initial_arrays_sha256']['x'] for r in results})==1
assert len({r['nvt_initial_arrays_sha256']['v'] for r in results})==3
result={'status':'passed','coordinate_arrays_identical':True,'velocity_arrays_distinct':True,'analysis_note':'Local GROMACS 2026.4 reanalysis compared against original Ubuntu 2026.3; original formal dynamics remain entirely on Ubuntu. Coordinate precision follows GRO 0.001nm. RMSD independent Bio.SVDSuperimposer; Q explicit pair loops at 0,8,11ns per trajectory.','runs':results}
(OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
