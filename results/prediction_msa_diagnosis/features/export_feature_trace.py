"""Pure preprocessing trace: no weights, no RunModel.predict, no network."""
from pathlib import Path
import hashlib
import json
from importlib.metadata import version
import numpy as np
from colabfold.batch import unserialize_msa, build_monomer_feature
from alphafold.model import config, features
from alphafold.model.tf import np_backend
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEQ = 'NLYIQWLKDGGPSSGRPPPS'
SEEDS = [42,2026,20261110]
conditions = ['query_only','deduplicated_saved_msa','raw_saved_msa','duplicate_query','duplicate_homolog']
records=[]
for condition in conditions:
    input_dir = ROOT / ('results/prediction_msa_diagnosis/followup/inputs' if condition.startswith('duplicate_') else 'results/prediction_ablation/inputs')
    text=(input_dir/f'{condition}.a3m').read_text()
    unpaired,paired,seqs,cardinality,templates=unserialize_msa([text],SEQ)
    raw=build_monomer_feature(SEQ,unpaired[0],templates[0])
    profile_input={'msa':raw['msa'].copy()}
    np_backend.fast_make_hhblits_profile(profile_input)
    profile=profile_input['hhblits_profile']
    np.savez_compressed(OUT/f'{condition}_raw_features.npz',msa=raw['msa'],hhblits_profile=profile,deletion_matrix_int=raw['deletion_matrix_int'])
    for variant in ['baseline','fixed_capacity_3','no_mask']:
        for seed in SEEDS:
            # ColabFold builds model_3_ptm runner without templates and swaps model_1 weights.
            cfg=config.model_config('model_3_ptm')
            cfg.data.eval.max_msa_clusters=3 if variant=='fixed_capacity_3' else len(raw['msa'])
            cfg.data.common.max_extra_msa=1
            cfg.data.common.num_recycle=3
            cfg.model.num_recycle=3
            cfg.data.eval.num_ensemble=1
            if variant=='no_mask':
                cfg.data.eval.masked_msa_replace_fraction=0.0
            processed=features.np_example_to_features(raw,cfg,random_seed=seed)
            selected={k:processed[k] for k in ['true_msa','bert_mask','msa_mask','msa_feat','extra_msa','extra_msa_mask'] if k in processed}
            destination=f'{condition}_{variant}_{seed}.npz'
            np.savez_compressed(OUT/destination,**selected)
            records.append({'condition':condition,'variant':variant,'seed':seed,'input_sha256':hashlib.sha256(text.encode()).hexdigest(),'preprocessed_rows':len(raw['msa']),'capacity':cfg.data.eval.max_msa_clusters,'replace_fraction':cfg.data.eval.masked_msa_replace_fraction,'output':destination,'shapes':{k:list(v.shape) for k,v in selected.items()},'masked_positions_by_ensemble':[[[int(i),int(j)+1] for i,j in np.argwhere(m>0)] for m in selected['bert_mask']],'extra_msa_mask_sum':float(selected['extra_msa_mask'].sum()),'true_msa_first_ensemble':selected['true_msa'][0].tolist(),'row_masks_by_ensemble':(selected['msa_mask'].max(-1)).tolist(),'query_msa_feature_sha256':hashlib.sha256(np.ascontiguousarray(selected['msa_feat'][:,0]).tobytes()).hexdigest()})
source_files=['colabfold/batch.py','colabfold/alphafold/models.py','alphafold/data/pipeline.py','alphafold/model/features.py','alphafold/model/tf/input_pipeline.py','alphafold/model/tf/data_transforms.py','alphafold/model/tf/np_backend.py','alphafold/model/tf/random_philox.py']
site=Path(__import__('colabfold').__file__).parent.parent
summary={'kind':'feature processing only; no model initialization, weights or inference','versions':{p:version(p) for p in ['colabfold','alphafold-colabfold','jax','numpy']},'source_sha256':{p:hashlib.sha256((site/p).read_bytes()).hexdigest() for p in source_files},'records':records}
(OUT/'trace_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for cond in conditions:
    d=np.load(OUT/f'{cond}_raw_features.npz')
    print(cond,'raw msa rows',len(d['msa']),'unique rows',len(np.unique(d['msa'],axis=0)),'column1 profile nonzero',[(int(i),float(v)) for i,v in enumerate(d['hhblits_profile'][0]) if v])
print('feature cases',len(records),'all extra_mask sums',[r['extra_msa_mask_sum'] for r in records])

comparisons=[]
for seed in SEEDS:
    for a,b,variant in [('raw_saved_msa','duplicate_homolog','baseline'),('raw_saved_msa','duplicate_query','baseline'),('raw_saved_msa','deduplicated_saved_msa','fixed_capacity_3')]:
        x=np.load(OUT/f'{a}_{variant}_{seed}.npz'); y=np.load(OUT/f'{b}_{variant}_{seed}.npz')
        comparisons.append({'seed':seed,'first':a,'second':b,'variant':variant,'all_saved_features_equal':all(np.array_equal(x[k],y[k]) for k in x.files),'query_bert_mask_equal':bool(np.array_equal(x['bert_mask'][:,0],y['bert_mask'][:,0])),'query_msa_feat_equal':bool(np.array_equal(x['msa_feat'][:,0],y['msa_feat'][:,0])),'all_bert_masks_equal':bool(np.array_equal(x['bert_mask'],y['bert_mask']))})
(OUT/'feature_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
no_mask=[]
for seed in SEEDS:
    x=np.load(OUT/f'raw_saved_msa_no_mask_{seed}.npz'); y=np.load(OUT/f'deduplicated_saved_msa_no_mask_{seed}.npz')
    no_mask.append({'seed':seed,'raw_dedup_query_feature_equal_when_mask_disabled':bool(np.array_equal(x['msa_feat'][:,0],y['msa_feat'][:,0])),'all_bert_masks_zero':bool(not x['bert_mask'].any() and not y['bert_mask'].any())})
(OUT/'no_mask_feature_comparison.json').write_text(json.dumps(no_mask,indent=2)+'\n')
