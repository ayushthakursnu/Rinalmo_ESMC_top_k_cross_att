#!/usr/bin/env python
"""
parpi_rinalmo_giga_esmc: frozen RiNALMo-giga (RNA) + frozen ESMC-600m (protein)
-> trainable projections -> top-k cross attention -> attention pooling -> 2-layer MLP.

Same data + split as PaRPI (one model per cell line, stratified 80/20 per RBP file).
Multi-seed (default 0 1 2), early stopping, best checkpoint by AUC.

Outputs in results/<cell>_<split_mode>/:
  epoch_metrics.csv       every epoch of every seed (train loss/AUC, val_* and test_* metrics, times)
  multiseed_summary.csv   best-AUC checkpoint per seed + mean / std / mean±std rows
  seed_<s>/best_model.pt, last.pt (resume), final_metrics.json, test_predictions.npz
  train.log

Example:
  python train.py --cell HeLa --seeds 0 1 2
"""
import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import CellLineData  # noqa: E402
from encoders import (ESMCEncoder, FakeProtEncoder, FakeRNAEncoder, RiNALMoEncoder,  # noqa: E402
                      build_protein_cache, build_rna_cache, rna_cache_dir)
from metrics import METRIC_KEYS, compute_metrics  # noqa: E402
from model import RNAProteinTopK  # noqa: E402

LOG = logging.getLogger("parpi_rg_esmc")


# ====================================================================== #
# args
# ====================================================================== #
def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("paths")
    g.add_argument("--parpi_root", default=os.path.join(HERE, "..", "PaRPI"),
                   help="PaRPI folder holding dataset/clip_data and dataset/protein")
    g.add_argument("--clip_subdir", default="dataset/clip_data")
    g.add_argument("--protein_subdir", default="dataset/protein")
    g.add_argument("--cache_dir", default=os.path.join(HERE, "cache"))
    g.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    g.add_argument("--run_name", default=None, help="default: <cell>_<split_mode>")

    g = ap.add_argument_group("data / split")
    g.add_argument("--cell", default="HeLa")
    g.add_argument("--split_mode", choices=["paper", "strict"], default="paper",
                   help="paper: PaRPI 80/20, best epoch picked on the 20%% (as PaRPI main.py). "
                        "strict: same 20%% test untouched, best epoch picked on a val slice of train")
    g.add_argument("--split_seed", type=int, default=1024,
                   help="seed that defines the PaRPI split (PaRPI default 1024). Fixed across training "
                        "seeds; use -1 to tie the split to each training seed instead")
    g.add_argument("--test_frac", type=float, default=0.2)
    g.add_argument("--val_frac", type=float, default=0.1, help="strict mode: fraction of train used as val")
    g.add_argument("--max_rna_len", type=int, default=1022)
    g.add_argument("--max_prot_len", type=int, default=0,
                   help="0 = full length. If >0, centre-crop cached protein features (GPU memory)")

    g = ap.add_argument_group("encoders / cache")
    g.add_argument("--rna_model", default="multimolecule/rinalmo-giga")
    g.add_argument("--prot_model", default="esmc_600m")
    g.add_argument("--enc_batch_size", type=int, default=64)
    g.add_argument("--esmc_window", type=int, default=2048)
    g.add_argument("--rna_in_ram", choices=["auto", "yes", "no"], default="auto",
                   help="load the RNA cache into RAM (fast) or read from memmap")
    g.add_argument("--ram_limit_gb", type=float, default=64.0)
    g.add_argument("--cache_only", action="store_true", help="build caches and exit")
    g.add_argument("--cache_part", choices=["all", "rna", "protein"], default="all",
                   help="build only one cache and exit (lets RiNALMo and ESMC live in different envs)")
    g.add_argument("--fake_encoders", action="store_true", help=argparse.SUPPRESS)

    g = ap.add_argument_group("model")
    g.add_argument("--hidden_size", type=int, default=256)
    g.add_argument("--num_heads", type=int, default=4)
    g.add_argument("--topk", type=int, default=32)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--l2norm", choices=["seq", "feature", "none"], default="seq",
                   help="seq = DeepMutDTA's F.normalize(dim=1) after projection")

    g = ap.add_argument_group("training")
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--batch_size", type=int, default=256)
    g.add_argument("--eval_batch_size", type=int, default=512)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--pos_weight", type=float, default=2.0, help="BCEWithLogits pos_weight (PaRPI uses 2)")
    g.add_argument("--max_epochs", type=int, default=100)
    g.add_argument("--patience", type=int, default=15, help="early stopping patience (epochs w/o AUC gain)")
    g.add_argument("--min_delta", type=float, default=1e-4)
    g.add_argument("--lr_patience", type=int, default=4, help="ReduceLROnPlateau patience")
    g.add_argument("--grad_clip", type=float, default=5.0)
    g.add_argument("--threshold", type=float, default=0.5)
    g.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--overwrite", action="store_true", help="retrain seeds that already finished")
    g.add_argument("--no_resume", action="store_true", help="ignore last.pt of unfinished seeds")
    g.add_argument("--per_rbp", action="store_true",
                   help="also write per_rbp_test_metrics.csv (per RBP file, best checkpoint)")
    g.add_argument("--aggregate_only", action="store_true", help="only rebuild multiseed_summary.csv")
    return ap.parse_args(argv)


# ====================================================================== #
# utils
# ====================================================================== #
def setup_logging(path):
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        h.setFormatter(fmt)
        LOG.addHandler(h)


def log(msg):
    LOG.info(msg)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def amp_dtype(args, device):
    if device.type != "cuda" or args.amp == "none":
        return None
    if args.amp == "fp16":
        return torch.float16
    if args.amp == "bf16" or torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def fmt_time(s):
    s = int(s)
    return "{:d}h{:02d}m{:02d}s".format(s // 3600, (s % 3600) // 60, s % 60)


# ====================================================================== #
# data pipeline over cached features
# ====================================================================== #
class RNAStore:
    """RNA embedding cache, either fully in RAM or as a lazily-opened memmap (worker safe)."""

    def __init__(self, emb_path, len_path, mode, ram_limit_gb):
        self.emb_path = emb_path
        self.lengths = np.load(len_path)
        size_gb = os.path.getsize(emb_path) / 1e9
        if mode == "yes" or (mode == "auto" and size_gb <= ram_limit_gb):
            t = time.time()
            self._arr = np.load(emb_path)          # fully in RAM
            log("[data] RNA cache loaded into RAM ({:.1f} GB, {:.0f}s)".format(size_gb, time.time() - t))
        else:
            self._arr = None
            log("[data] RNA cache read from memmap ({:.1f} GB > ram_limit_gb)".format(size_gb))
        self.dim = int(np.load(emb_path, mmap_mode="r").shape[-1])

    def arr(self):
        if self._arr is None:
            self._arr = np.load(self.emb_path, mmap_mode="r")
        return self._arr

    def __getstate__(self):  # never pickle a memmap into workers
        st = self.__dict__.copy()
        if isinstance(st.get("_arr"), np.memmap):
            st["_arr"] = None
        return st


class BatchDataset(Dataset):
    """Indexed by an array of sample ids; returns one ready-made batch."""

    def __init__(self, store, seq_idx, prot_idx, labels):
        self.store, self.seq_idx, self.prot_idx, self.labels = store, seq_idx, prot_idx, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, rows):
        rows = np.asarray(rows)
        s = self.seq_idx[rows]
        o = np.argsort(s, kind="stable")       # sorted rows -> sequential memmap reads
        rows, s = rows[o], s[o]
        lens = self.store.lengths[s]
        Lb = int(lens.max())
        x = np.ascontiguousarray(self.store.arr()[s, :Lb])
        return {"rna": torch.from_numpy(x), "len": torch.from_numpy(lens.astype(np.int64)),
                "prot": torch.from_numpy(self.prot_idx[rows]),
                "y": torch.from_numpy(self.labels[rows]), "rows": torch.from_numpy(rows)}


class BatchSampler_(Sampler):
    def __init__(self, idx, batch_size, shuffle, seed=0):
        self.idx, self.bs, self.shuffle, self.seed, self.epoch = np.asarray(idx), batch_size, shuffle, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        idx = self.idx
        if self.shuffle:
            idx = idx[np.random.RandomState(self.seed * 100003 + self.epoch).permutation(len(idx))]
        for b in range(0, len(idx), self.bs):
            yield idx[b:b + self.bs]

    def __len__(self):
        return (len(self.idx) + self.bs - 1) // self.bs


def make_loader(ds, idx, bs, shuffle, seed, workers):
    sampler = BatchSampler_(idx, bs, shuffle, seed)
    return DataLoader(ds, batch_size=None, sampler=sampler, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0,
                      prefetch_factor=4 if workers > 0 else None), sampler


class ProteinBank:
    """All cached protein features, padded, resident on the GPU."""

    def __init__(self, embs, device):
        self.lens = torch.tensor([e.shape[0] for e in embs], dtype=torch.long, device=device)
        P, D = int(self.lens.max()), embs[0].shape[1]
        bank = np.zeros((len(embs), P, D), dtype=np.float16)
        for i, e in enumerate(embs):
            bank[i, :e.shape[0]] = e
        self.bank = torch.from_numpy(bank).to(device)
        self.dim = D
        log("[data] protein bank: {} proteins, max_len={}, dim={} ({:.2f} GB on {})".format(
            len(embs), P, D, self.bank.numel() * 2 / 1e9, device))

    def gather(self, prot_idx):
        uniq, inv = torch.unique(prot_idx, return_inverse=True)
        lens = self.lens[uniq]
        Pb = int(lens.max())
        pu = self.bank[uniq, :Pb].float()
        pm = torch.arange(Pb, device=pu.device)[None, :] < lens[:, None]
        return pu, pm, inv


def to_device(batch, bank, device):
    rna = batch["rna"].to(device, non_blocking=True).float()
    lens = batch["len"].to(device, non_blocking=True)
    rmask = torch.arange(rna.shape[1], device=device)[None, :] < lens[:, None]
    prot = batch["prot"].to(device, non_blocking=True)
    pu, pm, inv = bank.gather(prot)
    y = batch["y"].to(device, non_blocking=True).float()
    return rna, rmask, pu, pm, inv, y


# ====================================================================== #
# train / eval
# ====================================================================== #
def run_epoch(model, loader, bank, criterion, device, adt, optimizer=None, scaler=None, grad_clip=5.0):
    train = optimizer is not None
    model.train(train)
    ys, ps, rows_all, loss_sum, n = [], [], [], 0.0, 0
    for batch in loader:
        rna, rmask, pu, pm, inv, y = to_device(batch, bank, device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device.type, dtype=adt, enabled=adt is not None):
                logits = model(rna, rmask, pu, pm, inv)
            loss = criterion(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
        loss_sum += loss.item() * len(y)
        n += len(y)
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logits.detach().float()).cpu().numpy())  # sigmoid only for metrics
        rows_all.append(batch["rows"].numpy())
    return loss_sum / max(n, 1), np.concatenate(ys), np.concatenate(ps), np.concatenate(rows_all)


def train_one_seed(seed, args, data, ds, bank, rna_dim, run_dir, epoch_csv, device):
    seed_dir = os.path.join(run_dir, "seed_{}".format(seed))
    os.makedirs(seed_dir, exist_ok=True)
    final_path = os.path.join(seed_dir, "final_metrics.json")
    if os.path.isfile(final_path) and not args.overwrite:
        log("[seed {}] already finished -> skipping (use --overwrite to retrain)".format(seed))
        return
    set_seed(seed)
    adt = amp_dtype(args, device)
    sel_name = data.selection_split()
    tr_idx, sel_idx, te_idx = data.indices("train"), data.indices(sel_name), data.indices("test")
    strict = sel_name == "val"

    train_loader, train_sampler = make_loader(ds, tr_idx, args.batch_size, True, seed, args.num_workers)
    sel_loader, _ = make_loader(ds, sel_idx, args.eval_batch_size, False, seed, args.num_workers)
    te_loader = make_loader(ds, te_idx, args.eval_batch_size, False, seed, args.num_workers)[0] if strict else None

    model = RNAProteinTopK(rna_dim=rna_dim, prot_dim=bank.dim, hidden_size=args.hidden_size,
                           num_heads=args.num_heads, topk=args.topk, dropout=args.dropout,
                           l2norm=args.l2norm).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=args.lr_patience)
    scaler = torch.amp.GradScaler("cuda") if adt == torch.float16 else None

    state = {"epoch": 0, "best_auc": -1.0, "best_epoch": 0, "bad_epochs": 0, "elapsed": 0.0}
    last_path = os.path.join(seed_dir, "last.pt")
    best_path = os.path.join(seed_dir, "best_model.pt")
    if os.path.isfile(last_path) and not args.no_resume and not args.overwrite:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        state = ck["state"]
        log("[seed {}] resuming after epoch {} (best AUC {:.4f} @ {})".format(
            seed, state["epoch"], state["best_auc"], state["best_epoch"]))
    _prune_epoch_csv(epoch_csv, seed, keep_upto=state["epoch"])

    log("[seed {}] trainable params: {:,} | train={} {}={} test={} | amp={}".format(
        seed, n_params, len(tr_idx), sel_name, len(sel_idx), len(te_idx), adt))

    stopped_early = False
    for epoch in range(state["epoch"] + 1, args.max_epochs + 1):
        if state["bad_epochs"] >= args.patience:      # resumed after early stop was reached
            stopped_early = True
            break
        t0 = time.time()
        train_sampler.set_epoch(epoch)
        tr_loss, ty, tp, _ = run_epoch(model, train_loader, bank, criterion, device, adt,
                                       optimizer, scaler, args.grad_clip)
        tr_m = compute_metrics(ty, tp, args.threshold)
        with torch.no_grad():
            sel_loss, sy, sp, _ = run_epoch(model, sel_loader, bank, criterion, device, adt)
            sel_m = compute_metrics(sy, sp, args.threshold)
            if strict:
                te_loss, ey, ep, _ = run_epoch(model, te_loader, bank, criterion, device, adt)
                te_m = compute_metrics(ey, ep, args.threshold)
            else:
                te_loss, te_m = sel_loss, sel_m
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(sel_m["AUC"])

        is_best = (sel_m["AUC"] > state["best_auc"] + args.min_delta) or not os.path.isfile(best_path)
        if is_best:
            state.update(best_auc=sel_m["AUC"], best_epoch=epoch, bad_epochs=0)
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args),
                        "rna_dim": rna_dim, "prot_dim": bank.dim}, best_path)
        else:
            state["bad_epochs"] += 1
        ep_time = time.time() - t0
        state["elapsed"] += ep_time
        state["epoch"] = epoch

        row = {"seed": seed, "epoch": epoch, "lr": lr_now, "train_loss": tr_loss,
               "train_AUC": tr_m["AUC"], "train_ACC": tr_m["ACC"], "val_loss": sel_loss}
        row.update({"val_" + k: sel_m[k] for k in METRIC_KEYS})
        row["test_loss"] = te_loss
        row.update({"test_" + k: te_m[k] for k in METRIC_KEYS})
        row.update({"is_best": int(is_best), "best_epoch_so_far": state["best_epoch"],
                    "best_val_AUC_so_far": state["best_auc"], "epoch_time_s": round(ep_time, 2),
                    "cum_time_s": round(state["elapsed"], 2), "selection_split": sel_name})
        _append_csv(epoch_csv, row)

        log("[seed {}] ep {:3d} | loss {:.4f} trAUC {:.4f} | {} AUC {:.4f} AUPR {:.4f} ACC {:.4f} F1 {:.4f} "
            "MCC {:.4f} | best {:.4f}@{} {}| {:.0f}s lr {:.1e}".format(
                seed, epoch, tr_loss, tr_m["AUC"], sel_name, sel_m["AUC"], sel_m["AUPR"], sel_m["ACC"],
                sel_m["F1"], sel_m["MCC"], state["best_auc"], state["best_epoch"],
                "* " if is_best else "", ep_time, lr_now))

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": state}, last_path)
        if state["bad_epochs"] >= args.patience:
            stopped_early = True
            log("[seed {}] early stopping at epoch {} (no AUC gain for {} epochs)".format(
                seed, epoch, args.patience))
            break

    # ---- final evaluation of the best-AUC checkpoint on the test split ----
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    te_loader_f = te_loader or make_loader(ds, te_idx, args.eval_batch_size, False, seed, args.num_workers)[0]
    with torch.no_grad():
        te_loss, ey, ep, erows = run_epoch(model, te_loader_f, bank, criterion, device, adt)
    te_m = compute_metrics(ey, ep, args.threshold)
    final = {"seed": seed, "best_epoch": ck["epoch"], "epochs_run": state["epoch"],
             "stopped_early": stopped_early, "selection_split": sel_name,
             "best_val_AUC": state["best_auc"], "train_time_s": round(state["elapsed"], 1),
             "avg_epoch_time_s": round(state["elapsed"] / max(state["epoch"], 1), 1),
             "test_loss": te_loss, "test": te_m, "n_test": int(len(ey)), "trainable_params": n_params}
    np.savez_compressed(os.path.join(seed_dir, "test_predictions.npz"), rows=erows, y=ey, prob=ep,
                        file_idx=data.file_idx[erows])
    with open(final_path, "w") as fh:
        json.dump(final, fh, indent=2)
    log("[seed {}] BEST epoch {} -> test AUC {:.4f} AUPR {:.4f} ACC {:.4f} F1 {:.4f} MCC {:.4f} TPR {:.4f} "
        "SCC {:.4f} PCC {:.4f} | train time {}".format(
            seed, ck["epoch"], te_m["AUC"], te_m["AUPR"], te_m["ACC"], te_m["F1"], te_m["MCC"],
            te_m["TPR"], te_m["SCC"], te_m["PCC"], fmt_time(state["elapsed"])))
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _append_csv(path, row):
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not os.path.isfile(path), index=False)


def _prune_epoch_csv(path, seed, keep_upto):
    """Drop rows of `seed` with epoch > keep_upto (fresh start or resume) so reruns never duplicate."""
    if not os.path.isfile(path):
        return
    df = pd.read_csv(path)
    keep = (df["seed"] != seed) | (df["epoch"] <= keep_upto)
    if not keep.all():
        df[keep].to_csv(path, index=False)


# ====================================================================== #
# multi-seed aggregate
# ====================================================================== #
def aggregate(run_dir, data=None, per_rbp=False, threshold=0.5):
    rows = []
    for d in sorted(os.listdir(run_dir)):
        f = os.path.join(run_dir, d, "final_metrics.json")
        if d.startswith("seed_") and os.path.isfile(f):
            with open(f) as fh:
                r = json.load(fh)
            row = {"seed": r["seed"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                   "stopped_early": r["stopped_early"], "selection_split": r["selection_split"],
                   "best_val_AUC": r["best_val_AUC"]}
            row.update({k: r["test"][k] for k in METRIC_KEYS})
            row.update({"n_test": r["n_test"], "train_time_s": r["train_time_s"],
                        "avg_epoch_time_s": r["avg_epoch_time_s"]})
            rows.append(row)
    if not rows:
        log("[summary] no finished seeds yet")
        return None
    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    num_cols = ["best_epoch", "epochs_run", "best_val_AUC"] + METRIC_KEYS + ["train_time_s", "avg_epoch_time_s"]
    mean = df[num_cols].mean()
    std = df[num_cols].std(ddof=1) if len(df) > 1 else df[num_cols].std(ddof=0)
    m_row = {"seed": "mean", **mean.to_dict()}
    s_row = {"seed": "std", **std.to_dict()}
    pm_row = {"seed": "mean±std"}
    for c in num_cols:
        f = "{:.4f} ± {:.4f}" if (c in METRIC_KEYS or c == "best_val_AUC") else "{:.1f} ± {:.1f}"
        pm_row[c] = f.format(mean[c], std[c])
    out = pd.concat([df, pd.DataFrame([m_row, s_row, pm_row])], ignore_index=True)
    out.insert(1, "n_seeds", len(df))
    path = os.path.join(run_dir, "multiseed_summary.csv")
    out.to_csv(path, index=False)
    log("[summary] {} seeds -> {}".format(len(df), path))
    log("[summary] test " + "  ".join("{} {:.4f}±{:.4f}".format(k, mean[k], std[k]) for k in METRIC_KEYS))

    if per_rbp and data is not None:
        prow = []
        for s in df["seed"]:
            z = np.load(os.path.join(run_dir, "seed_{}".format(s), "test_predictions.npz"))
            for fi in np.unique(z["file_idx"]):
                m = z["file_idx"] == fi
                r = {"seed": s, "data_name": data.data_names[fi], "rbp": data.rbp_of_file[fi], "n": int(m.sum())}
                r.update(compute_metrics(z["y"][m], z["prob"][m], threshold))
                prow.append(r)
        pdf = pd.DataFrame(prow)
        pdf.to_csv(os.path.join(run_dir, "per_rbp_test_metrics.csv"), index=False)
        log("[summary] per-RBP metrics -> per_rbp_test_metrics.csv")
    return out


# ====================================================================== #
# main
# ====================================================================== #
def main(args):
    run_name = args.run_name or "{}_{}".format(args.cell, args.split_mode)
    run_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(os.path.join(run_dir, "train.log"))
    log("=" * 100)
    log("args: " + json.dumps(vars(args)))
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    if args.split_seed < 0 and len(args.seeds) > 1:
        log("[warn] --split_seed -1: each training seed gets a different PaRPI-style split")

    def build_data(split_seed):
        return CellLineData(args.parpi_root, args.cell, split_mode=args.split_mode, split_seed=split_seed,
                            test_frac=args.test_frac, val_frac=args.val_frac, clip_subdir=args.clip_subdir,
                            protein_subdir=args.protein_subdir, max_rna_len=args.max_rna_len, log=log)

    data = build_data(args.split_seed if args.split_seed >= 0 else args.seeds[0])
    if args.aggregate_only:
        aggregate(run_dir, data, args.per_rbp, args.threshold)
        return
    data.save_split(os.path.join(run_dir, "split_seed{}.npz".format(data.split_seed)))
    data.verify_against_parpi(args.parpi_root, log=log)

    device = torch.device(args.device)
    # ---------------- caches (computed once, reused forever) ----------------
    rna_tag = args.rna_model if not args.fake_encoders else "fake_rna"
    prot_tag = args.prot_model if not args.fake_encoders else "fake_prot"
    rna_fac = (lambda: FakeRNAEncoder()) if args.fake_encoders else \
        (lambda: RiNALMoEncoder(args.rna_model, device, log=log))
    prot_fac = (lambda: FakeProtEncoder()) if args.fake_encoders else \
        (lambda: ESMCEncoder(args.prot_model, device, window=args.esmc_window, log=log))

    t = time.time()
    if args.cache_part in ("all", "rna"):
        emb_path, len_path = build_rna_cache(data.unique_seqs, rna_cache_dir(args.cache_dir, rna_tag, args.cell),
                                             rna_fac, batch_size=args.enc_batch_size, log=log)
    if args.cache_part in ("all", "protein"):
        prot_embs = build_protein_cache(data.unique_prots, data.prot_names, args.cache_dir, prot_tag, prot_fac,
                                        max_prot_len=args.max_prot_len or None, log=log)
    log("[cache] {} ready in {}".format(args.cache_part, fmt_time(time.time() - t)))
    if args.cache_only or args.cache_part != "all":
        return

    store = RNAStore(emb_path, len_path, args.rna_in_ram, args.ram_limit_gb)
    bank = ProteinBank(prot_embs, device)
    del prot_embs
    epoch_csv = os.path.join(run_dir, "epoch_metrics.csv")

    for seed in args.seeds:
        if args.split_seed < 0 and seed != data.split_seed:
            data = build_data(seed)   # caches are keyed by sequence content, so still reused
        ds = BatchDataset(store, data.seq_idx, data.prot_idx, data.labels)
        log("-" * 100)
        train_one_seed(seed, args, data, ds, bank, store.dim, run_dir, epoch_csv, device)
        aggregate(run_dir, data, args.per_rbp, args.threshold)   # refresh after every seed

    log("[done] per-epoch log : {}".format(epoch_csv))
    log("[done] multi-seed    : {}".format(os.path.join(run_dir, "multiseed_summary.csv")))


if __name__ == "__main__":
    main(get_args())
