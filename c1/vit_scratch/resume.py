"""Helpers to make sweeps resumable and to merge a results copy back.

Lives in c1/vit_scratch (Ammar's folder) and is also used by c2/peft, so no change to common/ is needed.
"""
import csv
import os
import re


def hw_of(notes):
    """The hw tag in a notes string ("hw=colab-t4 ..." -> "colab-t4"), or None."""
    if not isinstance(notes, str):
        return None
    m = re.search(r"hw=([\w.-]+)", notes)
    return m.group(1) if m else None


def missing_seeds(path, challenge, method, config_id, seeds, hw):
    """Seeds with no row yet for (challenge, method, config_id, hw) in the CSV at `path`."""
    if not os.path.exists(path):
        return list(seeds)
    with open(path, newline="") as f:
        done = {int(r["seed"]) for r in csv.DictReader(f)
                if r["challenge"] == challenge and r["method"] == method
                and r["config_id"] == config_id and hw_of(r["notes"]) == hw}
    return [s for s in seeds if s not in done]


def merge_results(src, dst):
    """Append the rows of results CSV `src` that `dst` doesn't have yet. Returns how many."""
    with open(dst, newline="") as f:
        have = {tuple(r) for r in csv.reader(f)}
    with open(src, newline="") as f:
        new = [r for r in list(csv.reader(f))[1:] if tuple(r) not in have]
    with open(dst, "a", newline="") as f:
        csv.writer(f).writerows(new)
    return len(new)
