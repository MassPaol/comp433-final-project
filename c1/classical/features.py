"""Hand-crafted feature extractors for 32x32 RGB images.

Each extractor maps uint8 images (N, 32, 32, 3) to a float32 matrix (N, D).
None of them learn anything from data: they are fixed functions, so they are
allowed in Challenge 1 (no pretrained models, no external data).
"""
import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog as _skimage_hog
from skimage.feature import local_binary_pattern

# Scattering operators keyed by (J, L, device). Emptied by clear_cache() at the
# start of every fit, so each fit pays the (small) cost of building the filters.
_SCATTERING_OPS = {}


def flip(X):
    """Horizontal flip of a batch of (N, H, W, C) images."""
    return X[:, :, ::-1]


def _gray_float(X):
    """(N, 32, 32) grayscale in [0, 1]."""
    return rgb2gray(X.astype(np.float32) / 255.0)


def raw(X, gray=False):
    """Pixels scaled to [0, 1], flattened. 3072 dims (1024 if gray)."""
    Z = _gray_float(X) if gray else X.astype(np.float32) / 255.0
    return Z.reshape(len(X), -1).astype(np.float32)


def hsv_hist(X, bins=16):
    """Per-image histograms of H, S and V, each normalized to sum 1. 3 * bins dims."""
    hsv = rgb2hsv(X.astype(np.float32) / 255.0).reshape(len(X), -1, 3)
    idx = np.minimum((hsv * bins).astype(np.int64), bins - 1)  # (N, P, 3) bin index
    n, p, _ = idx.shape
    out = np.empty((n, 3 * bins), dtype=np.float32)
    rows = np.repeat(np.arange(n), p)
    for c in range(3):
        counts = np.bincount(rows * bins + idx[:, :, c].ravel(), minlength=n * bins)
        out[:, c * bins:(c + 1) * bins] = counts.reshape(n, bins) / p
    return out


def hog(X, orientations=9, pixels_per_cell=4, cells_per_block=2, color=False):
    """Histogram of Oriented Gradients (Dalal & Triggs 2005), one vector per image.

    color=False runs on the grayscale image; color=True lets skimage take the
    channel with the strongest gradient at each pixel.
    """
    kw = dict(orientations=orientations,
              pixels_per_cell=(pixels_per_cell, pixels_per_cell),
              cells_per_block=(cells_per_block, cells_per_block),
              block_norm="L2-Hys", feature_vector=True)
    if color:
        imgs = X.astype(np.float32) / 255.0
        feats = [_skimage_hog(im, channel_axis=-1, **kw) for im in imgs]
    else:
        feats = [_skimage_hog(im, **kw) for im in _gray_float(X)]
    return np.asarray(feats, dtype=np.float32)


def lbp(X, P=8, R=1, grid=2):
    """Uniform Local Binary Pattern histograms (Ojala et al. 2002) on a grid x grid layout.

    (P + 2) bins per cell, each cell histogram normalized to sum 1.
    """
    gray = np.round(_gray_float(X) * 255).astype(np.uint8)
    n_bins = P + 2
    cell = gray.shape[1] // grid
    out = np.empty((len(X), grid * grid * n_bins), dtype=np.float32)
    for i, g in enumerate(gray):
        codes = local_binary_pattern(g, P, R, method="uniform").astype(np.int64)
        hists = []
        for r in range(grid):
            for c in range(grid):
                block = codes[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
                h = np.bincount(block.ravel(), minlength=n_bins).astype(np.float32)
                hists.append(h / block.size)
        out[i] = np.concatenate(hists)
    return out


def _scattering_op(J, L, device):
    key = (J, L, device)
    if key not in _SCATTERING_OPS:
        # Import the 2D torch frontend directly: `kymatio.torch` also imports the
        # 3D frontend, which needs scipy.special.sph_harm (removed in SciPy >= 1.17).
        from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D
        _SCATTERING_OPS[key] = ScatteringTorch2D(J=J, shape=(32, 32), L=L).to(device)
    return _SCATTERING_OPS[key]


def scattering(X, J=2, L=8, color=True, pool=1):
    """Wavelet scattering coefficients (Bruna & Mallat 2013) via kymatio.

    The filters are fixed Morlet wavelets, so nothing is learned or pretrained.
    Output per channel: (1 + J*L + L^2 * J*(J-1)/2) maps of size (32 / 2^J)^2,
    optionally average-pooled by `pool`. Runs on the GPU when one is available.
    """
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    op = _scattering_op(J, L, device)
    imgs = X.astype(np.float32) / 255.0
    imgs = imgs.transpose(0, 3, 1, 2) if color else _gray_float(X)[:, None]
    outs = []
    with torch.no_grad():
        for start in range(0, len(imgs), 256):
            batch = torch.from_numpy(np.ascontiguousarray(imgs[start:start + 256])).to(device)
            S = op(batch)  # (B, C, K, h, w)
            b, c, k, h, w = S.shape
            S = S.reshape(b, c * k, h, w)
            if pool > 1:
                S = F.avg_pool2d(S, pool)
            outs.append(S.reshape(b, -1).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def clear_cache():
    """Forget the built scattering operators."""
    _SCATTERING_OPS.clear()


FEATURES = {
    "raw": raw,
    "hsv_hist": hsv_hist,
    "hog": hog,
    "lbp": lbp,
    "scattering": scattering,
}


def extract(X, specs):
    """Concatenate features: specs is a list of (name, params) pairs."""
    parts = [FEATURES[name](X, **params) for name, params in specs]
    return np.concatenate(parts, axis=1).astype(np.float32)
