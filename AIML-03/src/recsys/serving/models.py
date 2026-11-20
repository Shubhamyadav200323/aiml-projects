"""Model backends for the online path.

`LocalModels` loads the exported SavedModels in-process. `TritonModels` sends the
same tensors to NVIDIA Triton Inference Server over HTTP. Both expose
`user_embedding(...)` and `rank(kind, ...)`, so the recommendation service is
unaware of where inference runs; set `TRITON_URL` to switch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class ModelBackend(Protocol):
    def user_embedding(self, inputs: dict[str, np.ndarray]) -> np.ndarray: ...
    def rank(self, kind: str, inputs: dict[str, np.ndarray]) -> np.ndarray: ...


class LocalModels:
    def __init__(self, models_dir: str | Path) -> None:
        from recsys.tf_compat import tf

        self.tf = tf
        d = Path(models_dir)
        self._user = tf.saved_model.load(str(d / "user_tower" / "1" / "model.savedmodel")).signatures["serving_default"]
        self._rankers = {}
        for p in sorted(d.glob("ranker_*")):
            sm = tf.saved_model.load(str(p / "1" / "model.savedmodel"))
            self._rankers[p.name.removeprefix("ranker_")] = sm.signatures["serving_default"]

    @property
    def ranker_kinds(self) -> list[str]:
        return list(self._rankers)

    def user_embedding(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        tf = self.tf
        out = self._user(user_idx=tf.constant(inputs["user_idx"]), user_dense=tf.constant(inputs["user_dense"]),
                         history=tf.constant(inputs["history"]))
        return out["embedding"].numpy()

    def rank(self, kind: str, inputs: dict[str, np.ndarray]) -> np.ndarray:
        tf = self.tf
        out = self._rankers[kind](user_idx=tf.constant(inputs["user_idx"]), item_idx=tf.constant(inputs["item_idx"]),
                                  dense=tf.constant(inputs["dense"]))
        return out["score"].numpy()


class TritonModels:  # pragma: no cover - requires a running Triton server
    def __init__(self, url: str, ranker_kinds: tuple[str, ...] = ("dcn_v2", "wide_deep")) -> None:
        import tritonclient.http as httpclient

        self.http = httpclient
        self.client = httpclient.InferenceServerClient(url=url, concurrency=8)
        self.ranker_kinds = list(ranker_kinds)

    def _infer(self, model: str, inputs: dict[str, np.ndarray], output: str) -> np.ndarray:
        dtype = {np.dtype("int64"): "INT64", np.dtype("float32"): "FP32"}
        reqs = []
        for name, arr in inputs.items():
            r = self.http.InferInput(name, list(arr.shape), dtype[arr.dtype])
            r.set_data_from_numpy(arr)
            reqs.append(r)
        res = self.client.infer(model, reqs, outputs=[self.http.InferRequestedOutput(output)])
        return res.as_numpy(output)

    def user_embedding(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        return self._infer("user_tower", inputs, "embedding")

    def rank(self, kind: str, inputs: dict[str, np.ndarray]) -> np.ndarray:
        return self._infer(f"ranker_{kind}", {k: inputs[k] for k in ("user_idx", "item_idx", "dense")}, "score")
