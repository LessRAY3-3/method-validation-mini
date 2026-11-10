#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/prediction/colabfold logs .cache/colabfold_jax
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR="$ROOT/.cache/colabfold_jax"

.colabfold/bin/colabfold_batch \
  data/prepared/1L2Y.fasta \
  results/prediction/colabfold \
  --msa-mode mmseqs2_uniref_env \
  --model-type alphafold2_ptm \
  --num-models 1 \
  --num-seeds 1 \
  --random-seed 20261110 \
  --num-recycle 3 \
  --rank plddt \
  --compile-mode fast \
  2>&1 | tee logs/colabfold.log

.conda/bin/python scripts/analyze_predictions.py
