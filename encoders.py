"""
Frozen encoders + on-disk embedding caches.

RNA     : RiNALMo-giga (MultiMolecule 'multimolecule/rinalmo-giga'; falls back to
          the native lbcb-sci/RiNALMo package 'giga-v1'). 1280-d per nucleotide.
Protein : ESMC-600m (Biohub/esm, 'esmc_600m'). 1152-d per residue.

Both encoders are FROZEN, so each unique sequence is embedded exactly once and
stored on disk. Special tokens (CLS/EOS) are stripped; only residue tokens are
kept so the cross-attention sees real positions only.

Cache layout (under --cache_dir):
  rna/<rna_model_tag>/<cell>/emb.npy        float16 memmap (N_unique, Lmax, 1280)
                             lengths.npy    int32  (N_unique,)
                             seqs.txt       one sequence per row (row order == emb rows)
                             progress.json  resumable: rows done; COMPLETE flag
  protein/<prot_model_tag>/<sha1>_<NAME>.npy  float16 (L, 1152), one per protein

The RNA cache is resumable: if generation is interrupted, re-running continues
from the last finished chunk instead of starting over.
"""
import gc
import hashlib
import json
import os
import time

import numpy as np
import torch

FP16_MAX = 65000.0


def _tag(name):
    return name.replace("/", "__")


def _seqs_hash(seqs):
    h = hashlib.sha1()
    for s in seqs:
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()


def _to_fp16(arr):
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=FP16_MAX, neginf=-FP16_MAX)
    return np.clip(arr, -FP16_MAX, FP16_MAX).astype(np.float16)


def _amp_dtype(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ====================================================================== #
# RNA encoder: RiNALMo-giga
# ====================================================================== #
class RiNALMoEncoder:
    def __init__(self, model_id="multimolecule/rinalmo-giga", device="cuda", log=print):
        self.device = torch.device(device)
        self.log = log
        self.backend = None
        import importlib.util
        if importlib.util.find_spec("multimolecule") is not None:
            # installed -> use it; any import error here is a real env problem and is raised as-is
            from multimolecule import RnaTokenizer, RiNALMoModel  # noqa: F401
            self.tok = RnaTokenizer.from_pretrained(model_id)
            self.model = RiNALMoModel.from_pretrained(model_id)
            self.backend = "multimolecule"
            self.dim = int(self.model.config.hidden_size)
        else:
            if importlib.util.find_spec("rinalmo") is None:
                raise ImportError("Neither 'multimolecule' nor 'rinalmo' is installed in this Python env")
            from rinalmo.pretrained import get_pretrained_model
            native = "giga-v1" if "giga" in model_id else model_id
            self.model, self.alphabet = get_pretrained_model(model_name=native)
            self.backend = "rinalmo"
            self.dim = None  # detected on first batch
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.amp = _amp_dtype(self.device)
        self._warned = False
        log("[encoder] RNA  : RiNALMo via {} ({}) frozen on {}".format(self.backend, model_id, self.device))

    @torch.inference_mode()
    def encode(self, seqs):
        """list[str] -> list[np.ndarray (len_i, D) float32] (special tokens removed)."""
        ctx = torch.autocast(self.device.type, dtype=self.amp) if self.amp else _null()
        out = []
        if self.backend == "multimolecule":
            enc = self.tok(list(seqs), return_tensors="pt", padding=True,
                           return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask", None)
            enc = {k: v.to(self.device) for k, v in enc.items()
                   if k in ("input_ids", "attention_mask")}
            with ctx:
                h = self.model(**enc).last_hidden_state.float()
            attn = enc["attention_mask"].bool()
            if special is not None:
                keep_mask = attn & (~special.bool().to(self.device))
            else:
                keep_mask = None
            for i, s in enumerate(seqs):
                L = len(s)
                if keep_mask is not None and int(keep_mask[i].sum()) == L:
                    out.append(h[i][keep_mask[i]].cpu().numpy())
                else:
                    if not self._warned:
                        self.log("[encoder] RNA: special-token mask mismatch; using positions 1..L")
                        self._warned = True
                    out.append(h[i, 1:1 + L].cpu().numpy())
        else:
            toks = torch.tensor(self.alphabet.batch_tokenize(list(seqs)), dtype=torch.int64,
                                device=self.device)
            with ctx:
                h = self.model(toks)["representation"].float()
            for i, s in enumerate(seqs):
                out.append(h[i, 1:1 + len(s)].cpu().numpy())
        if self.dim is None and out:
            self.dim = out[0].shape[-1]
        return out

    def close(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ====================================================================== #
# Protein encoder: ESMC-600m
# ====================================================================== #
class ESMCEncoder:
    def __init__(self, model_name="esmc_600m", device="cuda", window=2048, log=print):
        self.device = torch.device(device)
        self.window = window
        self.log = log
        self.backend = None
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        self._ESMProtein, self._LogitsConfig = ESMProtein, LogitsConfig
        client = None
        for kwargs in ({"device": self.device, "use_flash_attn": False},
                       {"device": self.device}, {}):
            try:
                client = ESMC.from_pretrained(model_name, **kwargs)
                break
            except TypeError:
                continue
        if client is None:
            raise RuntimeError("Could not load ESMC model '{}'".format(model_name))
        self.client = client.to(self.device).eval()
        for p in self.client.parameters():
            p.requires_grad_(False)
        self.backend = "esm.ESMC"
        log("[encoder] Prot : ESMC ({}) frozen on {}".format(model_name, self.device))

    @torch.inference_mode()
    def _embed_one(self, seq):
        pt = self.client.encode(self._ESMProtein(sequence=seq))
        out = self.client.logits(pt, self._LogitsConfig(sequence=True, return_embeddings=True))
        e = out.embeddings
        if e.dim() == 3:
            e = e[0]
        return e[1:-1].float().cpu().numpy()  # strip <cls> / <eos>

    def encode(self, seq):
        """Full-length per-residue embedding. Sequences longer than `window` are
        embedded with half-overlapping windows and overlapping positions averaged."""
        L = len(seq)
        if L <= self.window:
            return self._embed_one(seq)
        stride = self.window // 2
        starts = list(range(0, L - self.window + 1, stride))
        if starts[-1] + self.window < L:
            starts.append(L - self.window)
        acc, cnt = None, np.zeros((L, 1), dtype=np.float32)
        for s in starts:
            e = self._embed_one(seq[s:s + self.window])
            if acc is None:
                acc = np.zeros((L, e.shape[1]), dtype=np.float32)
            acc[s:s + self.window] += e
            cnt[s:s + self.window] += 1
        return acc / np.maximum(cnt, 1)

    def close(self):
        del self.client
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ====================================================================== #
# Fake encoders (debug/smoke test only: --fake_encoders)
# ====================================================================== #
class FakeRNAEncoder:
    dim = 32

    def __init__(self, *a, **k):
        g = np.random.RandomState(0)
        self.table = g.randn(128, self.dim).astype(np.float32)

    def encode(self, seqs):
        return [self.table[[ord(c) % 128 for c in s]] + 0.01 * np.arange(len(s))[:, None] for s in seqs]

    def close(self):
        pass


class FakeProtEncoder:
    dim = 24

    def __init__(self, *a, **k):
        g = np.random.RandomState(1)
        self.table = g.randn(128, self.dim).astype(np.float32)

    def encode(self, seq):
        return self.table[[ord(c) % 128 for c in seq]]

    def close(self):
        pass


# ====================================================================== #
# Cache builders
# ====================================================================== #
def rna_cache_dir(cache_root, rna_model, cell):
    return os.path.join(cache_root, "rna", _tag(rna_model), cell)


def build_rna_cache(seqs, cache_dir, encoder_factory, batch_size=64, log=print):
    """
    Embed every unique RNA sequence once into a float16 memmap. Resumable.
    Returns (memmap_path, lengths_path).
    """
    os.makedirs(cache_dir, exist_ok=True)
    emb_path = os.path.join(cache_dir, "emb.npy")
    len_path = os.path.join(cache_dir, "lengths.npy")
    seq_path = os.path.join(cache_dir, "seqs.txt")
    prog_path = os.path.join(cache_dir, "progress.json")
    shash = _seqs_hash(seqs)
    N = len(seqs)
    Lmax = max(len(s) for s in seqs)

    prog = {}
    if os.path.isfile(prog_path):
        with open(prog_path) as fh:
            prog = json.load(fh)
        if prog.get("seqs_sha1") != shash or not os.path.isfile(emb_path):
            log("[cache] RNA cache at {} was built for a different sequence set - rebuilding".format(cache_dir))
            prog = {}
    if prog.get("complete"):
        log("[cache] RNA embeddings found ({} seqs, dim={}) -> reusing {}".format(N, prog["dim"], emb_path))
        return emb_path, len_path

    # deterministic processing order: by length so batches have little padding
    order = np.argsort(np.array([len(s) for s in seqs]), kind="stable")
    chunk = batch_size * 16
    done_chunks = int(prog.get("done_chunks", 0))
    n_chunks = (N + chunk - 1) // chunk

    enc = encoder_factory()
    if done_chunks == 0:
        # probe dim, allocate
        dim = enc.encode([seqs[order[0]]])[0].shape[-1]
        gb = N * Lmax * dim * 2 / 1e9
        log("[cache] allocating RNA cache: {} seqs x {} nt x {} dims fp16 = {:.1f} GB".format(N, Lmax, dim, gb))
        emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(N, Lmax, dim))
        lengths = np.array([len(s) for s in seqs], dtype=np.int32)
        np.save(len_path, lengths)
        with open(seq_path, "w") as fh:
            fh.write("\n".join(seqs) + "\n")
        prog = {"seqs_sha1": shash, "n": N, "lmax": Lmax, "dim": int(dim), "done_chunks": 0,
                "complete": False}
        with open(prog_path, "w") as fh:
            json.dump(prog, fh)
    else:
        emb = np.load(emb_path, mmap_mode="r+")
        log("[cache] resuming RNA cache at chunk {}/{}".format(done_chunks, n_chunks))

    t0 = time.time()
    for c in range(done_chunks, n_chunks):
        rows = order[c * chunk:(c + 1) * chunk]
        for b in range(0, len(rows), batch_size):
            br = rows[b:b + batch_size]
            vecs = enc.encode([seqs[r] for r in br])
            for r, v in zip(br, vecs):
                emb[r, :v.shape[0]] = _to_fp16(v)
        emb.flush()
        prog["done_chunks"] = c + 1
        with open(prog_path, "w") as fh:
            json.dump(prog, fh)
        el = time.time() - t0
        rate = (c + 1 - done_chunks) / max(el, 1e-6)
        log("[cache] RNA chunk {}/{}  ({:.0f}s elapsed, ~{:.0f}s left)".format(
            c + 1, n_chunks, el, (n_chunks - c - 1) / max(rate, 1e-9)))
    prog["complete"] = True
    with open(prog_path, "w") as fh:
        json.dump(prog, fh)
    del emb
    enc.close()
    log("[cache] RNA cache complete: {}".format(emb_path))
    return emb_path, len_path


def protein_cache_path(cache_root, prot_model, name, seq):
    h = hashlib.sha1(seq.encode()).hexdigest()[:16]
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return os.path.join(cache_root, "protein", _tag(prot_model), "{}_{}.npy".format(h, safe))


def build_protein_cache(prot_seqs, prot_names, cache_root, prot_model, encoder_factory,
                        max_prot_len=None, log=print):
    """One .npy per protein (keyed by sequence hash). Returns list of arrays (float16)."""
    paths = [protein_cache_path(cache_root, prot_model, n, s) for n, s in zip(prot_names, prot_seqs)]
    missing = [i for i, p in enumerate(paths) if not os.path.isfile(p)]
    if missing:
        os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
        log("[cache] embedding {} / {} proteins with ESMC".format(len(missing), len(paths)))
        enc = encoder_factory()
        for k, i in enumerate(missing):
            t = time.time()
            e = enc.encode(prot_seqs[i])
            tmp = paths[i] + ".tmp.npy"
            np.save(tmp, _to_fp16(e))
            os.replace(tmp, paths[i])
            log("[cache]   {:>3}/{} {:<12} len={:<5d} {:.1f}s".format(
                k + 1, len(missing), prot_names[i], len(prot_seqs[i]), time.time() - t))
        enc.close()
    else:
        log("[cache] all {} protein embeddings found -> reusing".format(len(paths)))
    embs = []
    for p in paths:
        e = np.load(p)
        if max_prot_len and e.shape[0] > max_prot_len:
            # keep a centred crop if the user caps protein length for memory
            s = (e.shape[0] - max_prot_len) // 2
            e = e[s:s + max_prot_len]
        embs.append(e)
    return embs
