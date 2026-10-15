#!/usr/bin/env bash
# Two Python envs because RiNALMo (multimolecule, needs transformers 5) and ESMC
# (esm 3.2.x on Python 3.10, needs transformers < 4.48.2) cannot share one.
# Embeddings are just files in ./cache, so each env builds its own part once.
#
#   RNA_PY  : env with multimolecule        (default ../n_env_rinalmo)
#   MAIN_PY : env with esm, used for training (default ../n_env_rinalmo_esmc)
#
# Usage:  CELL=Hela bash parpi_rinalmo_giga_esmc_latest/run_hela.sh
#         SPLIT=strict CELL=Hela GPU=1 bash .../run_hela.sh
set -euo pipefail
cd "$(dirname "$0")"

CELL=${CELL:-Hela}
SPLIT=${SPLIT:-paper}
RNA_PY=${RNA_PY:-../n_env_rinalmo/bin/python}
MAIN_PY=${MAIN_PY:-../n_env_rinalmo_esmc/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p logs

# 1) RiNALMo-giga RNA embeddings (resumable; skipped instantly once complete)
"$RNA_PY" train.py --cell "$CELL" --cache_part rna 2>&1 | tee -a "logs/${CELL}_cache_rna.out"

# 2) ESMC-600m protein embeddings (skipped once complete)
"$MAIN_PY" train.py --cell "$CELL" --cache_part protein 2>&1 | tee -a "logs/${CELL}_cache_protein.out"

# 3) Train seeds 0 1 2 on cached features (no encoder is loaded here)
"$MAIN_PY" train.py --cell "$CELL" --split_mode "$SPLIT" --seeds 0 1 2 \
    --batch_size 256 --max_epochs 100 --patience 15 \
    2>&1 | tee -a "logs/${CELL}_${SPLIT}_train.out"
