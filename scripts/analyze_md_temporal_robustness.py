"""Exploratory temporal checks of completed MD time series; never simulates."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/md_deep_dive/temporal'
SEEDS=(20261110,20261111,20261112)
METRICS=('backbone_rmsd_common_reference_nm','q_ca_7A','q_ca_8A','q_ca_9A','sidechain_min_3_19_angstrom','sidechain_min_6_12_angstrom','sidechain_min_6_18_angstrom')
WINDOWS=((0,11),(0,1),(1,3),(3,8),(8,11),(7,11),(9,11))

def episodes(times, above):
    """Contiguous samples; no interpolation or fabricated crossing times."""
    times=np.asarray(times);above=np.asarray(above,dtype=bool)
    if len(times)!=len(above) or (len(times)>1 and not np.allclose(np.diff(times),10)):
        raise ValueError('Expected matching 10ps time grid')
    edges=np.diff(np.r_[False,above,False].astype(int))
    return [(int(times[a]),int(times[b-1]),int(b-a)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]

def block_means(times, values, start_ps, end_ps, width_ps):
    times=np.asarray(times); values=np.asarray(values)
    if (end_ps-start_ps)%width_ps or width_ps%10:
        raise ValueError('Complete 10ps-grid blocks required')
    result=[]
    for left in range(start_ps,end_ps,width_ps):
        selected=(times>=left)&(times<left+width_ps)
        if selected.sum()!=width_ps//10 or not np.array_equal(times[selected],np.arange(left,left+width_ps,10)):
            raise ValueError('Incomplete or duplicate block')
        result.append((left,float(values[selected].mean())))
    return result

def acf(values,max_lag=100):
    centered=np.asarray(values)-np.mean(values)
    denom=np.dot(centered,centered)
    if denom<=0: return np.full(min(max_lag+1,len(centered)),np.nan)
    return np.array([np.dot(centered[:len(centered)-lag],centered[lag:])/denom for lag in range(min(max_lag+1,len(centered)))])

def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    protocol=json.loads((OUT/'exploratory_protocol.json').read_text())
    for path,want in protocol['input_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=want:raise ValueError(f'Input changed: {path}')
    windows=[];blocks=[];thresholds=[];events=[];acf_rows=[];corr=[];series={}
    for seed in SEEDS:
        with (ROOT/f'results/md_replicates_gpu/aggregate_analysis_audited/seed_{seed}/timeseries.csv').open() as f:rows=list(csv.DictReader(f))
        times=np.array([int(r['time_ps']) for r in rows]); assert np.array_equal(times,np.arange(0,11001,10))
        vals={key:np.array([float(r[key]) for r in rows]) for key in METRICS}
        series[seed]=(times,vals)
        for a,b in WINDOWS:
            sel=(times>=a*1000)&(times<=b*1000 if b==11 else times<b*1000)
            for key,v in vals.items():
                windows.append(dict(seed=seed,window_ns=f'{a}-{b}',window_role=('original prespecified' if (a,b) in ((0,11),(8,11)) else 'posthoc sensitivity' if a in (7,9) else 'posthoc diagnostic'),metric=key,frames=int(sel.sum()),mean=float(v[sel].mean()),temporal_sd=float(v[sel].std()),min=float(v[sel].min()),max=float(v[sel].max())))
            for threshold in (5,6,7):
                above=vals['sidechain_min_3_19_angstrom'][sel]>threshold
                es=episodes(times[sel],above)
                thresholds.append(dict(seed=seed,window_ns=f'{a}-{b}',threshold_angstrom=threshold,sampled_fraction_above=float(above.mean()),n_samples_above=int(above.sum()),episodes=len(es),longest_sampled_span_ps=max((end-start for start,end,n in es),default=0)))
                for i,(start,end,n) in enumerate(es):events.append(dict(seed=seed,window_ns=f'{a}-{b}',threshold_angstrom=threshold,episode=i,start_sample_ps=start,last_sample_ps=end,n_samples=n,sampled_span_ps=end-start))
            for key in ('backbone_rmsd_common_reference_nm','q_ca_8A'):
                corr.append(dict(seed=seed,window_ns=f'{a}-{b}',metric_x='sidechain_min_3_19_angstrom',metric_y=key,pearson_descriptive=float(np.corrcoef(vals['sidechain_min_3_19_angstrom'][sel],vals[key][sel])[0,1]),note='Correlated temporal samples, no p-value; co-variation is not causality'))
        for start,end in ((0,11000),(8000,11000)):
            for width in (100,500,1000):
                for key,v in vals.items():
                    for left,mean in block_means(times,v,start,end,width):blocks.append(dict(seed=seed,range_ns=f'{start//1000}-{end//1000}',block_width_ps=width,start_ps=left,end_exclusive_ps=left+width,metric=key,mean=mean,frames=width//10))
        tail=(times>=8000)&(times<11000)
        for key in ('backbone_rmsd_common_reference_nm','q_ca_8A','sidechain_min_3_19_angstrom'):
            for lag,value in enumerate(acf(vals[key][tail])):acf_rows.append(dict(seed=seed,window_ns='8-11 excluding11ns',metric=key,lag_ps=lag*10,acf_biased=float(value)))
    for name,data in [('window_summary.csv',windows),('block_means.csv',blocks),('threshold_sensitivity.csv',thresholds),('threshold_episodes.csv',events),('tail_autocorrelation.csv',acf_rows),('descriptive_correlations.csv',corr)]:write(name,data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=('backbone_rmsd_common_reference_nm','q_ca_8A','sidechain_min_3_19_angstrom')
    labels=('Backbone RMSD (nm)','Geometric Q (8 Å)','Y3-P19 sidechain distance (Å)')
    fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for seed,(times,vals) in series.items():
        for ax,key,label in zip(axs,keys,labels):
            bs=block_means(times,vals[key],0,11000,100)
            ax.plot([(x+50)/1000 for x,y in bs],[y for x,y in bs],label=str(seed),lw=1.3)
            ax.set_ylabel(label);ax.axvspan(1,3,color='gray',alpha=.1);ax.axvspan(8,11,color='teal',alpha=.04)
    axs[0].legend(title='Velocity seed');axs[-1].set_xlabel('Production time (ns); nonoverlapping 0.1 ns block means')
    fig.suptitle('Exploratory temporal comparison; all three trajectories retained');fig.tight_layout();fig.savefig(OUT/'temporal_blocks.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for ax,key,label in zip(axs,keys,labels):
        for seed in SEEDS:
            rr=[r for r in acf_rows if r['seed']==seed and r['metric']==key]
            ax.plot([r['lag_ps']/1000 for r in rr],[r['acf_biased'] for r in rr],label=str(seed))
        ax.axhline(0,color='gray',lw=.7);ax.set(title=label,xlabel='Lag (ns)',ylabel='Normalized autocorrelation')
    axs[0].legend(fontsize=7);fig.suptitle('8-11 ns tail only (300 samples): descriptive, not a convergence test');fig.tight_layout();fig.savefig(OUT/'tail_autocorrelation.png',dpi=180);plt.close(fig)
    for path,want in protocol['input_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=want:raise ValueError(f'Input changed during analysis: {path}')
    summary={'status':'completed exploratory analysis; no new simulations','protocol_sha256':hashlib.sha256((OUT/'exploratory_protocol.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'n_trajectories':3,'rows':{'windows':len(windows),'blocks':len(blocks),'thresholds':len(thresholds),'episodes':len(events)},'limitations':['Windows and thresholds selected after seeing early deviation; descriptive exploration, not independent confirmation.','Blocks and frames are correlated, no frame-level tests or confidence intervals.','Tail autocorrelation uses only3ns; no stationary distribution, ESS or convergence inferred.','Thresholds are geometric cutoffs, not states or folding probabilities; sampled episodes miss events shorter than10ps.','Extending all three can test persistence but cannot undo common starting conformation or prove rare-event sampling.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
