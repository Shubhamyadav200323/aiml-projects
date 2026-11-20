"""Import TensorFlow in Keras-2 mode, which TensorFlow Recommenders requires.

Every TF module in this package imports `tf` and `tfrs` from here so the
environment flag is always set before TensorFlow initialises.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
import tensorflow_recommenders as tfrs  # noqa: E402

__all__ = ["tf", "tfrs"]
