"""Approximate nearest-neighbour index over item embeddings (FAISS).

* `flat`  - exact inner product (IndexFlatIP); best for <1M items.
* `hnsw`  - IndexHNSWFlat for large catalogues (sub-ms, ~99% recall).
* `ivf`   - IndexIVFFlat with `nlist` coarse clusters, `nprobe` at query time.

A NumPy brute-force fallback with the same API keeps unit tests and minimal
images working without the FAISS wheel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss

    HAS_FAISS = True
except Exception:  # pragma: no cover
    HAS_FAISS = False


class ItemIndex:
    def __init__(self, item_ids: list[str], kind: str = "flat", nlist: int = 64, nprobe: int = 16,
                 hnsw_m: int = 32) -> None:
        self.item_ids = list(item_ids)
        self.kind, self.nlist, self.nprobe, self.hnsw_m = kind, nlist, nprobe, hnsw_m
        self._index = None
        self._vectors: np.ndarray | None = None
        self.pos = {it: i for i, it in enumerate(self.item_ids)}

    @property
    def backend(self) -> str:
        return f"faiss-{self.kind}" if HAS_FAISS else "numpy"

    def build(self, vectors: np.ndarray) -> ItemIndex:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        assert len(v) == len(self.item_ids), "one vector per item"
        self._vectors = v
        if not HAS_FAISS:
            return self
        d = v.shape[1]
        if self.kind == "hnsw":
            idx = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efSearch = 128
        elif self.kind == "ivf":
            quant = faiss.IndexFlatIP(d)
            idx = faiss.IndexIVFFlat(quant, d, min(self.nlist, len(v)), faiss.METRIC_INNER_PRODUCT)
            idx.train(v)
            idx.nprobe = self.nprobe
        else:
            idx = faiss.IndexFlatIP(d)
        idx.add(v)
        self._index = idx
        return self

    def search(self, queries: np.ndarray, k: int,
               exclude: list[set[str]] | None = None) -> tuple[list[list[str]], np.ndarray]:
        """Top-k item ids per query, optionally excluding already-seen items."""
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        extra = max((len(e) for e in exclude), default=0) if exclude else 0
        kk = min(k + extra, len(self.item_ids))
        if self._index is not None:
            scores, idx = self._index.search(q, kk)
        else:
            sims = q @ self._vectors.T
            idx = np.argsort(-sims, axis=1)[:, :kk]
            scores = np.take_along_axis(sims, idx, 1)
        out_ids, out_scores = [], np.zeros((len(q), k), np.float32)
        for r in range(len(q)):
            ex = exclude[r] if exclude else set()
            ids, sc = [], []
            for j, s in zip(idx[r], scores[r]):
                if j < 0:
                    continue
                it = self.item_ids[j]
                if it not in ex:
                    ids.append(it)
                    sc.append(s)
                if len(ids) == k:
                    break
            out_ids.append(ids)
            out_scores[r, :len(sc)] = sc
        return out_ids, out_scores

    def vector(self, item_id: str) -> np.ndarray:
        return self._vectors[self.pos[item_id]]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "item_vectors.npy", self._vectors)
        meta = {"item_ids": self.item_ids, "kind": self.kind, "nlist": self.nlist, "nprobe": self.nprobe}
        (d / "index_meta.json").write_text(json.dumps(meta))
        if self._index is not None:
            faiss.write_index(self._index, str(d / "items.faiss"))

    @classmethod
    def load(cls, directory: str | Path) -> ItemIndex:
        d = Path(directory)
        meta = json.loads((d / "index_meta.json").read_text())
        idx = cls(meta["item_ids"], meta["kind"], meta["nlist"], meta["nprobe"])
        idx._vectors = np.load(d / "item_vectors.npy")
        if HAS_FAISS and (d / "items.faiss").exists():
            idx._index = faiss.read_index(str(d / "items.faiss"))
            if meta["kind"] == "ivf":
                idx._index.nprobe = meta["nprobe"]
        return idx
