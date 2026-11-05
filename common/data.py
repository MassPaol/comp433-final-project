"""Shared data loading.

DO NOT change `split_indices`. It is copied verbatim from the course testbed so
every method sees exactly the same problem instances.
"""
import os

import numpy as np

TUNE_SEEDS = list(range(0, 5))      # hyperparameter selection only
REPORT_SEEDS = list(range(10, 20))  # every number in the report comes from these

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")

DATA_ROOT = os.environ.get("COMP433_DATA", "./data")

_cache = None


def load_cifar_train(root=DATA_ROOT):
    """CIFAR-10 training set as (uint8 images (50000, 32, 32, 3), int labels (50000,))."""
    global _cache
    if _cache is None:
        from torchvision import datasets
        ds = datasets.CIFAR10(root=root, train=True, download=True)
        _cache = (ds.data, np.asarray(ds.targets))
    return _cache


def split_indices(targets, seed):
    # Verbatim from the testbed. Validation images also come from the CIFAR-10 *training* set.
    rng = np.random.RandomState(seed)
    classes = rng.permutation(10)[:2]
    targets = np.asarray(targets)
    train_ids, val_ids = [], []
    for label in classes:
        ids = rng.permutation(np.flatnonzero(targets == label))
        train_ids.extend(ids[:50])
        val_ids.extend(ids[50:400])
    return classes, np.array(train_ids), np.array(val_ids)


def load_instance(seed, root=DATA_ROOT):
    """One problem instance.

    Returns:
        X_train: uint8 (100, 32, 32, 3)   y_train: int64 (100,) in {0, 1}
        X_val:   uint8 (700, 32, 32, 3)   y_val:   int64 (700,) in {0, 1}
        classes: original CIFAR-10 labels of the two classes, e.g. array([3, 5])
                 -> names via CLASS_NAMES[classes[0]], CLASS_NAMES[classes[1]]

    Training images come sorted by class; shuffle inside your own `fit`.
    The validation images must NEVER be used for training (C1 or C2).
    """
    images, targets = load_cifar_train(root)
    classes, train_ids, val_ids = split_indices(targets, seed)
    assert not set(train_ids).intersection(val_ids)
    label_map = {int(c): i for i, c in enumerate(classes)}
    y_train = np.array([label_map[int(t)] for t in targets[train_ids]], dtype=np.int64)
    y_val = np.array([label_map[int(t)] for t in targets[val_ids]], dtype=np.int64)
    return images[train_ids], y_train, images[val_ids], y_val, classes
