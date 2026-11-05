"""Wall-clock timer that waits for queued GPU work before reading the clock."""
import time


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class Timer:
    """Usage:
        with Timer() as t:
            method.fit(X, y)
        t.seconds
    """

    def __enter__(self):
        _sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync()
        self.seconds = time.perf_counter() - self._t0
        return False


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
    except ImportError:
        pass
    return "cpu"
