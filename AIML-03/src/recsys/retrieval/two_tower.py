"""Two-tower retrieval model (TensorFlow Recommenders).

User tower : user-id embedding (with id dropout) + mean-pooled history of item
             embeddings (shared with the item tower) + dense taste features.
Item tower : item-id embedding + genre/popularity/quality features.
Both towers end in L2-normalised vectors, so FAISS inner-product search is
cosine similarity. Training uses in-batch sampled softmax with log-Q
popularity correction and accidental-hit removal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recsys.features.tables import HISTORY_LEN
from recsys.retrieval.inputs import ITEM_DENSE_DIM, USER_DENSE_DIM
from recsys.tf_compat import tf, tfrs


class _MLP(tf.keras.layers.Layer):
    def __init__(self, hidden: int, out: int, dropout: float, **kw):
        super().__init__(**kw)
        self.d1 = tf.keras.layers.Dense(hidden, activation="gelu")
        self.drop = tf.keras.layers.Dropout(dropout)
        self.d2 = tf.keras.layers.Dense(out)

    def call(self, x, training=False):
        return self.d2(self.drop(self.d1(x), training=training))


class UserTower(tf.keras.Model):
    def __init__(self, n_users: int, item_embedding: tf.keras.layers.Embedding, dim: int,
                 id_dropout: float = 0.3):
        super().__init__()
        self.user_emb = tf.keras.layers.Embedding(n_users + 1, dim)
        self.item_emb = item_embedding
        self.id_dropout = id_dropout
        self.mlp = _MLP(256, dim, 0.1)

    def call(self, inputs, training=False):
        u = self.user_emb(inputs["user_idx"])
        if training and self.id_dropout > 0:  # forces reliance on content -> better for sparse users
            keep = tf.cast(tf.random.uniform(tf.shape(u)[:1]) > self.id_dropout, u.dtype)[:, None]
            u = u * keep
        h = self.item_emb(inputs["history"])
        mask = tf.cast(inputs["history"] > 0, h.dtype)[..., None]
        pooled = tf.reduce_sum(h * mask, 1) / tf.maximum(tf.reduce_sum(mask, 1), 1.0)
        x = tf.concat([u, pooled, inputs["user_dense"]], axis=-1)
        return tf.math.l2_normalize(self.mlp(x, training=training), axis=-1)


class ItemTower(tf.keras.Model):
    def __init__(self, item_embedding: tf.keras.layers.Embedding, dim: int):
        super().__init__()
        self.item_emb = item_embedding
        self.mlp = _MLP(256, dim, 0.1)

    def call(self, inputs, training=False):
        x = tf.concat([self.item_emb(inputs["item_idx"]), inputs["item_dense"]], axis=-1)
        return tf.math.l2_normalize(self.mlp(x, training=training), axis=-1)


class TwoTowerModel(tfrs.Model):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, temperature: float = 0.05):
        super().__init__()
        item_emb = tf.keras.layers.Embedding(n_items + 1, dim, mask_zero=False)
        self.user_tower = UserTower(n_users, item_emb, dim)
        self.item_tower = ItemTower(item_emb, dim)
        self.task = tfrs.tasks.Retrieval(temperature=temperature, remove_accidental_hits=True)

    def compute_loss(self, features, training=False):
        q = self.user_tower({k: features[k] for k in ("user_idx", "user_dense", "history")}, training=training)
        c = self.item_tower({k: features[k] for k in ("item_idx", "item_dense")}, training=training)
        return self.task(q, c, candidate_sampling_probability=features["item_prob"],
                         candidate_ids=features["item_idx"], compute_metrics=False,
                         compute_batch_metrics=False)


@dataclass
class TrainConfig:
    dim: int = 64
    epochs: int = 8
    batch_size: int = 1024
    lr: float = 3e-3
    temperature: float = 0.05
    seed: int = 7


def train_two_tower(examples: dict[str, np.ndarray], n_users: int, n_items: int,
                    cfg: TrainConfig | None = None) -> tuple[TwoTowerModel, list[float]]:
    cfg = cfg or TrainConfig()
    tf.keras.utils.set_random_seed(cfg.seed)
    model = TwoTowerModel(n_users, n_items, cfg.dim, cfg.temperature)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.lr))
    ds = (tf.data.Dataset.from_tensor_slices(examples)
          .shuffle(len(examples["item_idx"]), seed=cfg.seed, reshuffle_each_iteration=True)
          .batch(cfg.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
    hist = model.fit(ds, epochs=cfg.epochs, verbose=0)
    return model, [float(x) for x in hist.history["loss"]]


def encode(tower: tf.keras.Model, inputs: dict[str, np.ndarray], batch: int = 4096) -> np.ndarray:
    n = len(next(iter(inputs.values())))
    out = [tower({k: v[i:i + batch] for k, v in inputs.items()}, training=False).numpy()
           for i in range(0, n, batch)]
    return np.vstack(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)


def export_user_tower(tower: UserTower, path: str | Path) -> None:
    """SavedModel with a fixed signature: loadable by Triton's tensorflow_savedmodel backend."""
    @tf.function(input_signature=[
        tf.TensorSpec([None], tf.int64, name="user_idx"),
        tf.TensorSpec([None, USER_DENSE_DIM], tf.float32, name="user_dense"),
        tf.TensorSpec([None, HISTORY_LEN], tf.int64, name="history"),
    ])
    def serve(user_idx, user_dense, history):
        return {"embedding": tower({"user_idx": user_idx, "user_dense": user_dense, "history": history})}

    tf.saved_model.save(tower, str(path), signatures={"serving_default": serve})


def export_item_tower(tower: ItemTower, path: str | Path) -> None:
    @tf.function(input_signature=[
        tf.TensorSpec([None], tf.int64, name="item_idx"),
        tf.TensorSpec([None, ITEM_DENSE_DIM], tf.float32, name="item_dense"),
    ])
    def serve(item_idx, item_dense):
        return {"embedding": tower({"item_idx": item_idx, "item_dense": item_dense})}

    tf.saved_model.save(tower, str(path), signatures={"serving_default": serve})
