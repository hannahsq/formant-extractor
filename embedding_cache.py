# embedding_cache.py
"""
Disk-based cache for Whisper embeddings with delta encoding + zstd compression.

Layout
------
<cache_root>/<model_slug>/<params_hash>/layer_00.npz
                                        layer_01.npz
                                        ...

Each .npz file contains two arrays saved with numpy's compressed format,
then further compressed with zstd at level 9:

  "delta"      : delta-encoded embeddings, float32, sorted by label
  "sort_index" : int32 argsort used to restore original sample order

Compression pipeline (encode)
------------------------------
1. Sort samples by label so adjacent rows are semantically similar.
2. Delta-encode along the sample axis in float32 space:
      delta[0]   = array[sort_index[0]]          (first row stored as-is)
      delta[i]   = array[sort_index[i]] - array[sort_index[i-1]]
   Adjacent embeddings for the same vowel are highly correlated, so
   deltas cluster tightly near zero — ideal for entropy coding.
3. Compress the resulting (N, ...) float32 array with zstd level 9.

This matches the filter chain "-m0=Delta -m1=zstd -mx=9" from 7-zip,
but with the delta computed in float32 space along the semantic axis
rather than the byte axis, which exploits the structure of embedding
data more directly.

Compression pipeline (decode)
------------------------------
1. Decompress with zstd.
2. Cumsum along axis 0 to recover the sorted array.
3. Apply the inverse permutation (argsort of sort_index) to restore
   original sample order.

Observed compression ratio: ~0.22-0.25 at ~400 MB/s decompression.

Fallback
--------
If zstandard is not installed, LZMA2 (stdlib lzma) is used instead,
giving ~0.35 CR at ~99 MB/s. Files written with one backend are not
readable by the other — the backend used is stored in a metadata key
inside each .npz file so load errors are caught cleanly.

Migration
---------
Existing caches written by the old per-sample .npy layout can be converted
with migrate_cache(cache_dir).
"""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import os
import shutil

import numpy as np
from tqdm import tqdm

try:
    import zstandard as zstd
    _ZSTD_AVAILABLE = True
except ImportError:
    import lzma
    _ZSTD_AVAILABLE = False

_FILE_EXT = ".npz"

# ---------------------------------------------------------------------------
# Delta encoding helpers
# ---------------------------------------------------------------------------

def _sort_and_delta_encode(
    array: np.ndarray,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort array along axis 0 and delta-encode.

    For float32 input, delta encoding uses uint32 wrapping arithmetic so
    the round-trip is bit-exact (same trick as 7-zip's Delta filter at the
    word level rather than byte level).

    For float16 input, delta encoding is skipped — the deltas between
    adjacent samples can be large enough that casting to f16 introduces
    per-step errors that compound through the cumsum. Instead the sorted
    f16 array is stored directly; zstd compresses small f16 values very
    efficiently anyway.

    Sorting is done by label first (when provided), then by per-sample mean
    as a tiebreaker within each label.

    Returns
    -------
    delta      : int32 view of uint32 deltas for f32; sorted f16 array for f16
    sort_index : (N,) int32 argsort used to restore original sample order
    """
    if labels is not None:
        sample_means = array.reshape(len(array), -1).mean(axis=1).astype(np.float64)
        sort_index   = np.lexsort((sample_means, labels)).astype(np.int32)
    else:
        sort_key   = array.reshape(len(array), -1).mean(axis=1)
        sort_index = np.argsort(sort_key).astype(np.int32)

    sorted_arr = array[sort_index]

    if array.dtype == np.float32:
        u32           = sorted_arr.view(np.uint32)
        delta_u32     = np.empty_like(u32)
        delta_u32[0]  = u32[0]
        delta_u32[1:] = u32[1:] - u32[:-1]   # wrapping uint32 subtraction
        # Cast to int32 for npz storage (same bit pattern, avoids unsigned issues)
        return delta_u32.view(np.int32), sort_index
    elif array.dtype == np.float16:
        # uint16 wrapping delta is worse than storing sorted f16 directly —
        # the deltas between adjacent f16 embeddings are not small enough in
        # integer space for zstd to exploit, whereas the label-sorted f16
        # values themselves cluster well. Return sorted array directly.
        return sorted_arr, sort_index
    else:
        raise ValueError(f"Unsupported dtype for delta encoding: {array.dtype}")


def _delta_decode_and_unsort(
    delta: np.ndarray,
    sort_index: np.ndarray,
) -> np.ndarray:
    """Invert _sort_and_delta_encode exactly."""
    if delta.dtype == np.int32:
        # float32 path: cumsum in uint32 wrapping space
        u32    = delta.view(np.uint32)
        out    = np.empty_like(u32)
        np.cumsum(u32, axis=0, out=out)
        recovered = out.view(np.float32)
    elif delta.dtype == np.float16:
        # float16 path: no delta applied, just unsort
        recovered = delta
    else:
        raise ValueError(f"Unrecognised delta dtype: {delta.dtype}")

    unsort = np.empty_like(sort_index)
    unsort[sort_index] = np.arange(len(sort_index), dtype=np.int32)
    return recovered[unsort]


# ---------------------------------------------------------------------------
# Compression backends
# ---------------------------------------------------------------------------

def _compress(array: np.ndarray, labels: np.ndarray | None = None) -> bytes:
    """
    Quantise to float16, delta-encode, serialise, and compress a layer array.

    The original dtype is stored as metadata so _decompress can cast back
    transparently. In practice arrays arrive as float32 and are returned as
    float32, with float16 used only on disk.
    """
    orig_dtype = np.dtype(array.dtype).str.encode()   # e.g. b'<f4' for float32
    array_f16  = array.astype(np.float16)
    delta, sort_index = _sort_and_delta_encode(array_f16, labels)

    buf = io.BytesIO()
    np.savez(buf, delta=delta, sort_index=sort_index,
             orig_dtype=np.frombuffer(orig_dtype, dtype=np.uint8))
    raw = buf.getvalue()

    if _ZSTD_AVAILABLE:
        cctx = zstd.ZstdCompressor(level=9)
        return b"zstd" + cctx.compress(raw)
    else:
        return b"lzma" + lzma.compress(raw, format=lzma.FORMAT_XZ,
                                        filters=[{"id": lzma.FILTER_LZMA2,
                                                  "dict_size": 1 << 18}])


def _decompress(data: bytes) -> np.ndarray:
    """
    Decompress, delta-decode, and restore original dtype.

    Legacy files written without orig_dtype metadata are returned as float32.
    """
    magic, payload = data[:4], data[4:]

    if magic == b"zstd":
        if not _ZSTD_AVAILABLE:
            raise RuntimeError(
                "Cache was written with zstd but zstandard is not installed. "
                "Run: pip install zstandard"
            )
        dctx = zstd.ZstdDecompressor()
        raw  = dctx.decompress(payload)
    elif magic == b"lzma":
        raw = lzma.decompress(payload, format=lzma.FORMAT_XZ)
    else:
        raise ValueError(f"Unrecognised compression magic bytes: {magic!r}")

    npz   = np.load(io.BytesIO(raw))
    array = _delta_decode_and_unsort(npz["delta"], npz["sort_index"])

    # Restore original dtype; default to float32 for legacy files
    if "orig_dtype" in npz:
        dtype = np.dtype(npz["orig_dtype"].tobytes().decode())
    else:
        dtype = np.float32

    return array.astype(dtype)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def cache_dir_for(
    cache_root: str,
    model_name: str,
    params: dict,
) -> str:
    """
    Return the cache directory for a given model + extraction config.
    The directory is deterministic: identical params always map to the
    same path.
    """
    key = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()[:12]
    model_slug = model_name.replace("/", "_")
    return os.path.join(cache_root, model_slug, key)


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def save_layer(
    cache_dir: str,
    layer_idx: int,
    array: np.ndarray,
    labels: np.ndarray | None = None,
):
    """
    Persist one layer's full embedding array, quantised to float16,
    delta-encoded, and compressed.

    Parameters
    ----------
    cache_dir : cache directory for this extraction config
    layer_idx : encoder layer index
    array     : (N, T, D) or (N, D) float32 array
    labels    : (N,) integer label array — when provided, samples are sorted
                by label before delta encoding so same-vowel rows are adjacent,
                improving compression ratio by ~5-10%.
    """
    _ensure_dir(cache_dir)
    path = os.path.join(cache_dir, f"layer_{layer_idx:02d}.npz")
    with open(path, "wb") as f:
        f.write(_compress(array, labels))


def load_embeddings(cache_dir: str) -> list[list[np.ndarray]] | None:
    """
    Load all cached layer embeddings, decompressing each layer in turn.

    Only files matching the exact pattern layer_NN.npz are loaded.
    Raises ValueError if the layer indices are not contiguous from 0,
    which catches accidental backup files left in the cache directory.

    Returns
    -------
    [layer][sample] -> (T, D) or (D,) array, or None if cache missing.
    """
    if not os.path.isdir(cache_dir):
        return None

    import re
    _LAYER_RE = re.compile(r'^layer_(\d{2})\.npz$')

    layer_files = {}
    for fname in os.listdir(cache_dir):
        m = _LAYER_RE.match(fname)
        if m:
            layer_files[int(m.group(1))] = fname

    if not layer_files:
        return None

    indices = sorted(layer_files)
    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError(
            f"Non-contiguous layer files in {cache_dir}: "
            f"found indices {indices}, expected {expected}. "
            f"Remove any backup or extra .npz files from the cache directory."
        )

    all_layers = []
    for idx in indices:
        path = os.path.join(cache_dir, layer_files[idx])
        with open(path, "rb") as f:
            array = _decompress(f.read())
        all_layers.append([array[i] for i in range(len(array))])

    return all_layers


def load_layer(cache_dir: str, layer_idx: int) -> list[np.ndarray] | None:
    """
    Load a single layer without decompressing the others.
    Useful for the probe sweep which processes one layer at a time.

    Returns
    -------
    list of per-sample arrays, or None if the file is missing.
    """
    path = os.path.join(cache_dir, f"layer_{layer_idx:02d}.npz")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        array = _decompress(f.read())
    return [array[i] for i in range(len(array))]


def is_cache_valid(cached: list[list[np.ndarray]], expected_n: int) -> bool:
    """
    Return True iff every layer has exactly expected_n samples.
    """
    for layer_idx, layer in enumerate(cached):
        if len(layer) != expected_n:
            print(
                f"Cache incomplete at layer {layer_idx}: "
                f"found {len(layer)} samples, expected {expected_n}. "
                f"Cache will be ignored."
            )
            return False
    return True


def n_cached_layers(cache_dir: str) -> int:
    """Return the number of valid layer_NN.npz files in cache_dir."""
    if not os.path.isdir(cache_dir):
        return 0
    import re
    _LAYER_RE = re.compile(r'^layer_(\d{2})\.npz$')
    return sum(1 for f in os.listdir(cache_dir) if _LAYER_RE.match(f))


# ---------------------------------------------------------------------------
# Two-pass cache construction
# ---------------------------------------------------------------------------

def save_sample(
    temp_dir: str,
    sample_idx: int,
    layer_arrays: list[np.ndarray],
):
    """
    Write one sample's embeddings for all layers to a temporary file.

    Called once per sample during Pass 1 (extraction). The file holds a
    single (n_layers, T, D) float16 array compressed with zstd, so each
    sample's RAM can be freed immediately after this call returns.

    Parameters
    ----------
    temp_dir     : directory for temporary per-sample files (must exist)
    sample_idx   : index of this sample in the dataset
    layer_arrays : list of (T, D) float32 arrays, one per encoder layer
    """
    stacked = np.stack(layer_arrays, axis=0).astype(np.float16)  # (L, T, D)
    buf = io.BytesIO()
    np.save(buf, stacked)
    if _ZSTD_AVAILABLE:
        cctx = zstd.ZstdCompressor(level=3)   # fast — these are temp files
        data = b"zstd" + cctx.compress(buf.getvalue())
    else:
        data = b"lzma" + lzma.compress(buf.getvalue(), format=lzma.FORMAT_XZ,
                                        filters=[{"id": lzma.FILTER_LZMA2,
                                                  "dict_size": 1 << 16}])
    path = os.path.join(temp_dir, f"{sample_idx:06d}.npz")
    with open(path, "wb") as f:
        f.write(data)


def _load_sample(path: str) -> np.ndarray:
    """Load a per-sample temp file, returning (L, T, D) float16 array."""
    with open(path, "rb") as f:
        data = f.read()
    magic, payload = data[:4], data[4:]
    if magic == b"zstd":
        if not _ZSTD_AVAILABLE:
            raise RuntimeError(
                "Sample file was written with zstd but zstandard is not installed."
            )
        raw = zstd.ZstdDecompressor().decompress(payload)
    elif magic == b"lzma":
        raw = lzma.decompress(payload, format=lzma.FORMAT_XZ)
    else:
        raise ValueError(f"Unrecognised magic: {magic!r}")
    return np.load(io.BytesIO(raw))


def consolidate_samples(
    temp_dir: str,
    cache_dir: str,
    labels: list[str],
    delete_temp: bool = True,
) -> None:
    """
    Pass 2: read per-sample temp files one layer at a time, stack into
    (N, T, D) arrays, and write compressed layer_NN.npz files.

    Peak memory during this pass is one full layer: (N, T, D) float16.

    Parameters
    ----------
    temp_dir    : directory containing per-sample .npz files from save_sample
    cache_dir   : destination for the final layer_NN.npz files
    labels      : dataset labels in original sample order, used for sort key
    delete_temp : if True, remove temp_dir after successful consolidation
    """
    import re, shutil

    sample_files = sorted(
        f for f in os.listdir(temp_dir)
        if re.match(r'^\d{6}\.npz$', f)
    )
    if not sample_files:
        raise ValueError(f"No sample files found in {temp_dir}")

    n_samples = len(sample_files)

    # Peek at the first file to learn shape — (L, T, D) for sequences, (L, D) for pooled
    first    = _load_sample(os.path.join(temp_dir, sample_files[0]))
    n_layers = first.shape[0]
    item_shape = first.shape[1:]   # (T, D) or (D,)
    del first

    _ensure_dir(cache_dir)

    unique_labels = sorted(set(labels))
    label_map  = {l: i for i, l in enumerate(unique_labels)}
    label_ints = np.array([label_map[l] for l in labels], dtype=np.int32)

    print(f"Consolidating {n_samples} samples × {n_layers} layers "
          f"(item shape {item_shape}) → {cache_dir}")
    for layer_idx in tqdm(range(n_layers), desc="Consolidating layers"):
        layer_buf = np.empty((n_samples, *item_shape), dtype=np.float16)
        for sample_pos, fname in enumerate(sample_files):
            sample_data = _load_sample(os.path.join(temp_dir, fname))
            layer_buf[sample_pos] = sample_data[layer_idx]
            del sample_data

        save_layer(cache_dir, layer_idx, layer_buf, labels=label_ints)
        del layer_buf

    if delete_temp:
        shutil.rmtree(temp_dir)
        print(f"Removed temp directory {temp_dir}")

    print(f"Consolidation complete → {cache_dir}")


def iter_layers(cache_dir: str) -> "Generator[tuple[int, list[np.ndarray]]]":
    """
    Yield (layer_idx, samples) one layer at a time without keeping more
    than one layer's worth of data in memory.

    Parameters
    ----------
    cache_dir : path to a completed cache directory with layer_NN.npz files

    Yields
    ------
    (layer_idx, list_of_per_sample_arrays)
        Each array is (T, D) float32 — dtype restored from float16 on disk.
    """
    import re
    _LAYER_RE = re.compile(r'^layer_(\d{2})\.npz$')
    layer_files = sorted(
        (int(m.group(1)), fname)
        for fname in os.listdir(cache_dir)
        if (m := _LAYER_RE.match(fname))
    )
    if not layer_files:
        raise ValueError(f"No layer files found in {cache_dir}")

    for layer_idx, fname in layer_files:
        path = os.path.join(cache_dir, fname)
        with open(path, "rb") as f:
            array = _decompress(f.read())          # (N, T, D) float32
        yield layer_idx, [array[i] for i in range(len(array))]
        del array                                   # explicit GC hint

def migrate_cache(cache_dir: str, delete_originals: bool = False) -> int:
    """
    Convert an old per-sample .npy cache to the compressed per-layer format.

    Old layout:  <cache_dir>/layer_NN/<sample_idx:04d>.npy
    New layout:  <cache_dir>/layer_NN.npz

    Parameters
    ----------
    cache_dir        : path to the existing cache directory
    delete_originals : if True, remove the old layer_NN/ subdirectories
                       after successful conversion. Defaults to False so
                       you can verify the output before deleting.

    Returns
    -------
    Number of layers successfully migrated.

    Raises
    ------
    FileNotFoundError if cache_dir does not exist.
    ValueError if no old-format layer directories are found.
    """
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    layer_dirs = sorted(
        d for d in os.listdir(cache_dir)
        if os.path.isdir(os.path.join(cache_dir, d))
        and d.startswith("layer_")
    )
    if not layer_dirs:
        raise ValueError(
            f"No old-format layer directories found in {cache_dir}. "
            f"Already migrated, or wrong directory?"
        )

    migrated = 0
    for ld in layer_dirs:
        layer_path = os.path.join(cache_dir, ld)
        files = sorted(f for f in os.listdir(layer_path) if f.endswith(".npy"))
        if not files:
            print(f"  {ld}: empty, skipping")
            continue

        print(f"  {ld}: loading {len(files)} samples...", end=" ", flush=True)
        arrays  = [np.load(os.path.join(layer_path, f)) for f in files]
        stacked = np.stack(arrays, axis=0)   # (N, T, D) or (N, D)

        layer_idx = int(ld.split("_")[1])
        save_layer(cache_dir, layer_idx, stacked)

        size_before = sum(
            os.path.getsize(os.path.join(layer_path, f)) for f in files
        )
        size_after = os.path.getsize(
            os.path.join(cache_dir, f"layer_{layer_idx:02d}.npz")
        )
        backend = "zstd" if _ZSTD_AVAILABLE else "lzma"
        ratio   = size_after / size_before if size_before else 0
        print(
            f"compressed [{backend}] -> layer_{layer_idx:02d}.npz  "
            f"({size_before / 1e6:.1f} MB -> {size_after / 1e6:.1f} MB, "
            f"ratio {ratio:.2f})",
            flush=True,
        )

        if delete_originals:
            shutil.rmtree(layer_path)

        migrated += 1

    print(f"\nMigration complete: {migrated} layer(s) converted in {cache_dir}")
    return migrated
