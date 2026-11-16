"""Dataset loading. Deliberately offline: the dataset ships with scikit-learn,
so a clone of this repo reproduces every number without a network call."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits

SEED = 1337


def load_split(seed: int = SEED):
    """Return (X_train, y_train, X_test, y_test) with a fixed, documented split.

    Pixels are 0..16 integers; we scale to 0..1 floats. No mean subtraction,
    which keeps every activation in the network non-negative and makes the
    quantization story easier to reason about.
    """
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_test = 360  # ~20%
    return X[n_test:], y[n_test:], X[:n_test], y[:n_test]
