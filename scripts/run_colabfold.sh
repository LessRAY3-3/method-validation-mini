#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-help}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.cache/matplotlib}"
# The only mode allowed to fetch model weights is prepare_weights. Resolve the
# official platform cache path; COLABFOLD_DATA_DIR can select a portable cache.
if [[ "$MODE" == prepare_weights || "$MODE" == predict ]]; then
  DATA_DIR="${COLABFOLD_DATA_DIR:-$("$ROOT/.colabfold/bin/python" -c 'from colabfold.download import default_data_dir; print(default_data_dir)')}"
fi
if [[ "$MODE" == prepare_weights ]]; then
  printf 'Preparing AlphaFold2 weights (network download only if absent): %s\n' "$DATA_DIR"
  "$ROOT/.colabfold/bin/python" - "$DATA_DIR" <<'PYCODE'
import sys
from pathlib import Path
from colabfold.download import download_alphafold_params
cache = Path(sys.argv[1])
download_alphafold_params("alphafold2_ptm", cache)
weights = cache / "params/params_model_1_ptm.npz"
if not weights.is_file() or weights.stat().st_size == 0:
    raise SystemExit("Weights missing or incomplete despite download marker; inspect the cache before retrying.")
print(f"Weights ready: {weights}")
PYCODE
  exit 0
elif [[ "$MODE" == analyze ]]; then
  INPUT="${2:-$ROOT/results/prediction/colabfold}"
  OUTPUT="${3:-$ROOT/results/prediction/analysis}"
  exec "$ROOT/.conda/bin/python" scripts/analyze_predictions.py --esm-scale 0-1 --colabfold-dir "$INPUT" --output-dir "$OUTPUT"
elif [[ "$MODE" != predict || $# -lt 2 ]]; then
  printf 'Usage:\n  bash scripts/run_colabfold.sh prepare_weights  # first-time network preparation\n  bash scripts/run_colabfold.sh analyze [prediction_directory] [analysis_directory]\n  bash scripts/run_colabfold.sh predict NEW_output_directory [saved_MSA.a3m]\nPrediction uses the saved MSA (no online MSA search). Existing output directories are rejected.\n'
  [[ "$MODE" == help ]] && exit 0
  exit 2
fi
OUTPUT="$2"
MSA="${3:-$ROOT/results/prediction/colabfold/1L2Y_model1.a3m}"
if [[ -e "$OUTPUT" ]]; then
  printf 'Refusing existing output directory: %s\n' "$OUTPUT" >&2
  exit 2
fi
[[ -s "$MSA" && "$MSA" == *.a3m ]] || { printf 'A saved, nonempty .a3m MSA is required\n' >&2; exit 2; }
# Fail before ColabFold can download weights; this mode is intentionally offline.
[[ -s "$DATA_DIR/params/params_model_1_ptm.npz" && -f "$DATA_DIR/params/download_finished.txt" ]] || {
  printf 'Existing AlphaFold2 weights required. Run bash scripts/run_colabfold.sh prepare_weights first (downloads weights), or set COLABFOLD_DATA_DIR to their parent directory.\n' >&2; exit 2;
}
mkdir -p "$OUTPUT" "$ROOT/.cache/colabfold_jax"
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR="$ROOT/.cache/colabfold_jax"
"$ROOT/.colabfold/bin/python" "$ROOT/.colabfold/bin/colabfold_batch" "$MSA" "$OUTPUT" \
  --data "$DATA_DIR" --model-type alphafold2_ptm --num-models 1 --num-seeds 1 \
  --random-seed 20261110 --num-recycle 3 --rank plddt --compile-mode fast \
  2>&1 | tee "$OUTPUT/run.log"
"$ROOT/.conda/bin/python" scripts/analyze_predictions.py --esm-scale 0-1 \
  --colabfold-dir "$OUTPUT" --output-dir "$OUTPUT/analysis"
"$ROOT/.conda/bin/python" scripts/prediction_provenance.py "$OUTPUT" "$MSA" \
  --weights "$DATA_DIR/params/params_model_1_ptm.npz"
