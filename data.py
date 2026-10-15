"""
PaRPI dataset loading + the exact PaRPI train/test split.

PaRPI (ljquanlab/PaRPI, main.py --train) builds ONE model per cell line from all
`dataset/clip_data/*_{cell}.tsv` files. For every file, in sorted() order, it
calls utils.split_dataset(), which does a stratified 80/20 split:

    np.random.seed(seed)                       # fix_seed(args.seed), default 1024
    for file in sorted(glob('*_{cell}.tsv')):
        ind0 = where(label <  0.5); ind1 = where(label >= 0.5)
        n_neg = int(len(ind0) * 0.2); n_pos = int(len(ind1) * 0.2)
        shuf_neg = np.random.permutation(len(ind0))   # neg first
        shuf_pos = np.random.permutation(len(ind1))   # then pos
        train = ind1[shuf_pos[n_pos:]] + ind0[shuf_neg[n_neg:]]
        test  = ind1[shuf_pos[:n_pos]] + ind0[shuf_neg[:n_neg]]

Nothing else touches the main-process NumPy RNG between those calls (RNAplfold
reseeding in rna_graph.py runs inside a multiprocessing Pool), so replaying the
same RandomState stream reproduces PaRPI's split exactly.

split_mode:
  paper  -> train 80% / test 20%; checkpoint selection on the 20% (what PaRPI's
            main.py does). Use this to compare with PaRPI's published numbers.
  strict -> same 20% test kept untouched; a stratified `val_frac` slice of the
            80% train is used for checkpoint selection / early stopping.
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

# PaRPI TSV columns (utils/utils.py::read_csv)
COL_TYPE, COL_LOC, COL_SEQ, COL_STR, COL_SCORE, COL_PROT, COL_LABEL = range(7)


def normalize_rna(seq):
    """Uppercase, DNA->RNA alphabet (RiNALMo/RnaTokenizer expect U)."""
    return str(seq).strip().upper().replace("T", "U")


def read_parpi_tsv(path):
    """Mirror of PaRPI utils.read_csv (header rows whose col0 == 'Type' dropped)."""
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    df = df.loc[df[COL_TYPE] != "Type"].reset_index(drop=True)
    seqs = df[COL_SEQ].to_numpy()
    prots = df[COL_PROT].to_numpy()
    labels = df[COL_LABEL].to_numpy().astype(np.float32)
    return seqs, prots, labels


def read_protein_fasta(path):
    seq = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith(">"):
                seq.append(line)
    return "".join(seq).upper().replace("*", "")


def list_cell_files(clip_dir, cell):
    files = sorted(glob.glob(os.path.join(clip_dir, "*_{}.tsv".format(cell))))
    if not files:
        raise FileNotFoundError(
            "No '*_{}.tsv' files found in {} - check --parpi_root / --cell".format(cell, clip_dir))
    return files


def parpi_split_indices(labels, rng, valid_frac=0.2):
    """One PaRPI split_dataset() call. Consumes rng exactly like PaRPI."""
    ind0 = np.where(labels < 0.5)[0]
    ind1 = np.where(labels >= 0.5)[0]
    n_neg = int(len(ind0) * valid_frac)
    n_pos = int(len(ind1) * valid_frac)
    shuf_neg = rng.permutation(len(ind0))
    shuf_pos = rng.permutation(len(ind1))
    train = np.concatenate((ind1[shuf_pos[n_pos:]], ind0[shuf_neg[n_neg:]]))
    test = np.concatenate((ind1[shuf_pos[:n_pos]], ind0[shuf_neg[:n_neg]]))
    return train, test


def _stratified_carve(idx, labels, frac, rng):
    """Split idx into (keep, carved) with the carved part stratified by label."""
    pos = idx[labels[idx] >= 0.5]
    neg = idx[labels[idx] < 0.5]
    pos = pos[rng.permutation(len(pos))]
    neg = neg[rng.permutation(len(neg))]
    n_pos, n_neg = int(len(pos) * frac), int(len(neg) * frac)
    carved = np.concatenate((pos[:n_pos], neg[:n_neg]))
    keep = np.concatenate((pos[n_pos:], neg[n_neg:]))
    return keep, carved


class CellLineData:
    """
    All samples of one cell line, flattened across RBP files.

    Attributes (all length N_samples unless noted):
      seq_idx    -> row in self.unique_seqs (RNA embedding cache row)
      prot_idx   -> row in self.unique_prots (protein embedding cache)
      labels     -> float32 {0,1}
      file_idx   -> index into self.files / self.data_names
      split      -> 'train' | 'val' | 'test'
    """

    def __init__(self, parpi_root, cell, split_mode="paper", split_seed=1024,
                 test_frac=0.2, val_frac=0.1, clip_subdir="dataset/clip_data",
                 protein_subdir="dataset/protein", max_rna_len=1022, log=print):
        self.cell = cell
        self.split_mode = split_mode
        self.split_seed = split_seed
        clip_dir = os.path.join(parpi_root, clip_subdir)
        prot_dir = os.path.join(parpi_root, protein_subdir)
        self.files = list_cell_files(clip_dir, cell)
        self.data_names = [os.path.splitext(os.path.basename(f))[0] for f in self.files]

        rng = np.random.RandomState(split_seed)          # == np.random.seed(seed)
        val_rng = np.random.RandomState(split_seed + 7919)  # independent stream (strict mode)

        seq_to_row, prot_to_row = {}, {}
        self.unique_seqs, self.unique_prots, self.prot_names = [], [], []
        seq_idx, prot_idx, labels, file_idx, split = [], [], [], [], []
        self.test_seqs_by_file = {}
        self.n_truncated = 0

        for fi, path in enumerate(self.files):
            seqs, prots, y = read_parpi_tsv(path)
            tr, te = parpi_split_indices(y, rng, valid_frac=test_frac)   # PaRPI split
            sp = np.empty(len(y), dtype=object)
            sp[tr] = "train"
            sp[te] = "test"
            if split_mode == "strict":
                _, va = _stratified_carve(tr, y, val_frac, val_rng)
                sp[va] = "val"
            elif split_mode != "paper":
                raise ValueError("split_mode must be 'paper' or 'strict'")

            # protein: PaRPI uses prot.iloc[0] of each file -> dataset/protein/<name>.fasta
            pname = str(prots[0]).strip()
            fasta = os.path.join(prot_dir, "{}.fasta".format(pname))
            if not os.path.isfile(fasta):
                raise FileNotFoundError("Protein FASTA missing for {}: {}".format(pname, fasta))
            pseq = read_protein_fasta(fasta)
            if pseq not in prot_to_row:
                prot_to_row[pseq] = len(self.unique_prots)
                self.unique_prots.append(pseq)
                self.prot_names.append(pname)
            p_row = prot_to_row[pseq]

            for s in seqs:
                s = normalize_rna(s)
                if len(s) > max_rna_len:
                    s = s[:max_rna_len]
                    self.n_truncated += 1
                r = seq_to_row.get(s)
                if r is None:
                    r = len(self.unique_seqs)
                    seq_to_row[s] = r
                    self.unique_seqs.append(s)
                seq_idx.append(r)
            prot_idx.append(np.full(len(y), p_row, dtype=np.int64))
            labels.append(y)
            file_idx.append(np.full(len(y), fi, dtype=np.int64))
            split.append(sp)
            self.test_seqs_by_file[self.data_names[fi]] = [normalize_rna(seqs[i]) for i in te]

        self.seq_idx = np.asarray(seq_idx, dtype=np.int64)
        self.prot_idx = np.concatenate(prot_idx)
        self.labels = np.concatenate(labels).astype(np.float32)
        self.file_idx = np.concatenate(file_idx)
        self.split = np.concatenate(split)
        self.rbp_of_file = [self.prot_names[self.prot_idx[self.file_idx == fi][0]]
                            for fi in range(len(self.files))]

        if self.n_truncated:
            log("[data] {} RNA sequences truncated to {} nt".format(self.n_truncated, max_rna_len))
        log(self.describe())

    # ------------------------------------------------------------------ #
    def indices(self, name):
        return np.where(self.split == name)[0]

    def selection_split(self):
        """Split used for best-epoch selection / early stopping."""
        return "val" if self.split_mode == "strict" else "test"

    def describe(self):
        lines = ["[data] cell={} files(RBPs)={} samples={} unique_rna={} unique_proteins={}".format(
            self.cell, len(self.files), len(self.labels), len(self.unique_seqs), len(self.unique_prots))]
        for name in ("train", "val", "test"):
            idx = self.indices(name)
            if len(idx):
                lines.append("[data]   {:<5} n={:<8d} pos={:<8d} neg={}".format(
                    name, len(idx), int(self.labels[idx].sum()), int((self.labels[idx] < 0.5).sum())))
        lines.append("[data] split_mode={} split_seed={} (selection on '{}')".format(
            self.split_mode, self.split_seed, self.selection_split()))
        return "\n".join(lines)

    def fingerprint(self):
        h = hashlib.sha1()
        h.update(json.dumps([self.split_mode, self.split_seed]).encode())
        h.update(self.split.astype("U5").tobytes())
        return h.hexdigest()[:12]

    def save_split(self, out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, split=self.split.astype("U5"), labels=self.labels,
                            file_idx=self.file_idx, data_names=np.array(self.data_names),
                            split_mode=self.split_mode, split_seed=self.split_seed)

    def verify_against_parpi(self, parpi_root, log=print):
        """
        If PaRPI's own run left dataset/test_set/<name>.npz (it saves test_seq),
        check our test sequences match. Only meaningful when PaRPI was run with
        the same seed as --split_seed (PaRPI default 1024).
        """
        tdir = os.path.join(parpi_root, "dataset", "test_set")
        checked, matched = 0, 0
        for name, ours in self.test_seqs_by_file.items():
            f = os.path.join(tdir, name + ".npz")
            if not os.path.isfile(f):
                continue
            try:
                theirs = np.load(f, allow_pickle=True)["test_seq"]
            except Exception:
                continue
            theirs = [normalize_rna(s) for s in theirs]
            checked += 1
            if sorted(theirs) == sorted(ours):
                matched += 1
        if checked:
            log("[split-check] {}/{} RBP files match PaRPI's saved test_set exactly".format(matched, checked))
        else:
            log("[split-check] no PaRPI dataset/test_set/*.npz found - skipped (split replayed from seed)")
        return checked, matched
