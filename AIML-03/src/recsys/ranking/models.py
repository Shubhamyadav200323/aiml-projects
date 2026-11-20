"""Deep ranking models that re-score retrieval candidates.

* **DCN-v2** (Wang et al., 2021): stacked full-rank cross layers
  (`tfrs.layers.dcn.Cross`) learn explicit bounded-degree feature interactions,
  in parallel with a deep MLP.
* **Wide & Deep** (Cheng et al., 2016): a linear "wide" part over dense and
  hand-crossed features memorises, a deep MLP over embeddings generalises.

Both expose the same inputs/outputs so they are interchangeable behind the
A/B experiment and in Triton.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recsys.ranking.features import RANK_DENSE_DIM
from recsys.tf_compat import tf, tfrs

_L2 = tf.keras.regularizers.l2(1e-5)


class _Base(tfrs.Model):
    """Shared input block. User identity is deliberately *not* embedded: the ranker
    sees users through their taste features, so it generalises to users with few
    labels instead of memorising the labelling window."""

    def __init__(self, n_users: int, n_items: int, emb_dim: int):
        super().__init__()
        self.item_emb = tf.keras.layers.Embedding(n_items + 1, emb_dim, embeddings_regularizer=_L2)
        self.norm = tf.keras.layers.LayerNormalization()
        self.task = tfrs.tasks.Ranking(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                       metrics=[tf.keras.metrics.AUC(name="auc", from_logits=True)])

    def _embed(self, inputs, training):
        return tf.concat([self.item_emb(inputs["item_idx"]), self.norm(inputs["dense"])], axis=-1)

    def compute_loss(self, features, training=False):
        logits = self({k: features[k] for k in ("user_idx", "item_idx", "dense")}, training=training)
        return self.task(labels=features["label"], predictions=logits)


class DCNRanker(_Base):
    name_ = "dcn_v2"

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 8, n_cross: int = 2,
                 deep: tuple[int, ...] = (128, 64)):
        super().__init__(n_users, n_items, emb_dim)
        self.cross = [tfrs.layers.dcn.Cross(projection_dim=None, kernel_initializer="glorot_uniform",
                                            kernel_regularizer=_L2) for _ in range(n_cross)]
        self.deep = tf.keras.Sequential([tf.keras.layers.Dense(u, activation="relu", kernel_regularizer=_L2)
                                         for u in deep] + [tf.keras.layers.Dropout(0.2)])
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x0 = self._embed(inputs, training)
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return tf.squeeze(self.out(tf.concat([x, self.deep(x0, training=training)], -1)), -1)


class WideDeepRanker(_Base):
    name_ = "wide_deep"

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 8, deep: tuple[int, ...] = (128, 64)):
        super().__init__(n_users, n_items, emb_dim)
        self.wide = tf.keras.layers.Dense(1)
        self.deep = tf.keras.Sequential([tf.keras.layers.Dense(u, activation="relu", kernel_regularizer=_L2)
                                         for u in deep] + [tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(1)])

    def call(self, inputs, training=False):
        dense = inputs["dense"]
        # wide part: raw dense + explicit crosses of the top interaction signals
        cross = dense[:, 0:1] * dense[:, 2:3]  # retrieval_score x genre_match
        wide = self.wide(tf.concat([dense, cross], -1))
        deep = self.deep(self._embed(inputs, training), training=training)
        return tf.squeeze(wide + deep, -1)


RANKERS = {"dcn_v2": DCNRanker, "wide_deep": WideDeepRanker}


@dataclass
class RankerConfig:
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 11


def train_ranker(kind: str, data: dict[str, np.ndarray], n_users: int, n_items: int,
                 val: dict[str, np.ndarray] | None = None, cfg: RankerConfig | None = None):
    cfg = cfg or RankerConfig()
    tf.keras.utils.set_random_seed(cfg.seed)
    model = RANKERS[kind](n_users, n_items)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.lr))
    ds = (tf.data.Dataset.from_tensor_slices(data).shuffle(len(data["label"]), seed=cfg.seed)
          .batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE))
    vds = tf.data.Dataset.from_tensor_slices(val).batch(8192) if val is not None else None
    hist = model.fit(ds, epochs=cfg.epochs, validation_data=vds, verbose=0)
    return model, {k: [float(x) for x in v] for k, v in hist.history.items()}


def predict(model, inputs: dict[str, np.ndarray]) -> np.ndarray:
    return tf.sigmoid(model({k: inputs[k] for k in ("user_idx", "item_idx", "dense")}, training=False)).numpy()


def export_ranker(model, path: str | Path) -> None:
    @tf.function(input_signature=[
        tf.TensorSpec([None], tf.int64, name="user_idx"),
        tf.TensorSpec([None], tf.int64, name="item_idx"),
        tf.TensorSpec([None, RANK_DENSE_DIM], tf.float32, name="dense"),
    ])
    def serve(user_idx, item_idx, dense):
        return {"score": tf.sigmoid(model({"user_idx": user_idx, "item_idx": item_idx, "dense": dense}))}

    tf.saved_model.save(model, str(path), signatures={"serving_default": serve})
