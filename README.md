# parpi_rinalmo_giga_esmc

Frozen **RiNALMo-giga** (RNA, 1280-d) + frozen **ESMC-600m** (protein, 1152-d) → trainable
projections → **top-k cross attention** (DeepMutDTA `ExactTopKAttention`, both directions,
shared weights) → **attention pooling** → concat → **2-layer MLP** → raw logit
(`BCEWithLogitsLoss`, sigmoid only for metrics).

## Folder layout expected

```
Learning/
├── PaRPI/                       # untouched: dataset/clip_data/*_<cell>.tsv, dataset/protein/*.fasta
└── parpi_rinalmo_giga_esmc/     # this folder
    ├── train.py  data.py  encoders.py  model.py  metrics.py  run_hela.sh
    ├── cache/                   # embeddings (created automatically, reused forever)
    └── results/<cell>_<split>/  # CSVs, checkpoints, logs
```
Nothing is written inside `PaRPI/`. Point elsewhere with `--parpi_root`.

## Run

```bash
cd Learning/parpi_rinalmo_giga_esmc
bash run_hela.sh                                  # HeLa, seeds 0 1 2, paper split
# or directly
python train.py --cell HeLa --seeds 0 1 2
```

## Data split (same as PaRPI)

PaRPI trains one model per cell line on all `*_<cell>.tsv` files and, for each file in sorted
order, does a stratified 80/20 split with `np.random.seed(1024)` → `permutation(neg)` →
`permutation(pos)`. `data.py` replays that RNG stream exactly (verified against PaRPI's own
`split_dataset` code). The split is fixed across training seeds (`--split_seed 1024`), so
seed-to-seed variance reflects training only.

* `--split_mode paper` (default): best epoch chosen on the 20% held-out set — exactly what
  PaRPI's `main.py` does, so numbers are comparable to the paper.
* `--split_mode strict`: same 20% test is never used for selection; a stratified 10% of train
  (`--val_frac`) is used for early stopping/best-AUC selection. Recommended for the journal
  submission alongside the paper-mode numbers.

If `PaRPI/dataset/test_set/*.npz` exists from an original PaRPI run, the script checks the
test sequences match and logs `[split-check] x/y RBP files match`.

## Embedding cache

* RNA: `cache/rna/multimolecule__rinalmo-giga/<cell>/emb.npy` — float16 memmap of every
  *unique* RNA, special tokens stripped. Resumable chunk-by-chunk (`progress.json`).
  Size ≈ N_unique × 101 × 1280 × 2 B (≈ 26 GB per 100k sequences). Loaded into RAM when
  smaller than `--ram_limit_gb` (64), otherwise read from the memmap.
* Protein: `cache/protein/esmc_600m/<sha1>_<RBP>.npy`, full length (proteins > 2048 aa are
  embedded in overlapping windows and averaged). Kept on the GPU during training.
* Caches are keyed by sequence content, so seeds, split modes and reruns all reuse them.

## Outputs (`results/<cell>_<split_mode>/`)

| file | content |
|---|---|
| `epoch_metrics.csv` | one row per epoch per seed: lr, train loss/AUC/ACC, `val_*` and `test_*` AUC, AUPR, ACC, F1, MCC, TPR, TNR, Precision, SCC, PCC, `is_best`, epoch & cumulative time |
| `multiseed_summary.csv` | best-AUC checkpoint of each seed evaluated on test + `mean`, `std` (ddof=1) and `mean±std` rows |
| `seed_<s>/best_model.pt` | best-AUC checkpoint |
| `seed_<s>/last.pt` | resume point (rerun the same command after a crash) |
| `seed_<s>/final_metrics.json`, `test_predictions.npz` | per-seed final numbers and raw test probabilities |
| `train.log`, `config.json`, `split_seed1024.npz` | logs, args, exact split |

`--per_rbp` additionally writes `per_rbp_test_metrics.csv` (PaRPI reports per-RBP AUCs).
`--aggregate_only` rebuilds the summary, e.g. after running seeds on different GPUs.

SCC / PCC are Spearman / Pearson correlation between predicted probability and the 0/1 label.
Threshold metrics (ACC, F1, MCC, TPR) use `--threshold 0.5`.

## Useful knobs

`--batch_size` (lower if OOM on long RBPs), `--max_prot_len 1500` (centre-crop protein
features), `--hidden_size 256 --num_heads 4 --topk 32`, `--lr 3e-4`, `--pos_weight 2`
(PaRPI's value), `--patience 15`, `--max_epochs 100`, `--l2norm seq|feature|none`.
