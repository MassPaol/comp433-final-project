"""The interface every C1 and C2 method implements.

fit/predict take raw CIFAR images. Do ALL preprocessing inside them
(normalization, resizing to 224, feature extraction, augmentation), so the
final-evaluation script can run any method without knowing how it works.
"""


class Method:
    name = "base"

    def __init__(self, **hparams):
        self.hparams = hparams

    def fit(self, X, y):
        """X: uint8 (N, 32, 32, 3); y: int (N,) in {0, 1}. Returns self.

        Settings like the number of epochs must be fixed in advance:
        no early stopping on validation data.
        """
        raise NotImplementedError

    def predict(self, X):
        """X: uint8 (N, 32, 32, 3). Returns int array (N,) in {0, 1}."""
        raise NotImplementedError

    def n_trainable_params(self):
        """Optional: number of trainable parameters, logged in the results table."""
        return None

    def extra_notes(self):
        """Optional: extra values for this run's notes column, e.g. {"extract_s": 0.41}.

        Called after `predict`; the harness appends them as "key=value".
        """
        return {}
