# dataset.py
"""
Audio loading and preprocessing utilities for local datasets.

For HuggingFace datasets, see datasets_hf.py.

Personal dataset layout
-----------------------
load_personal_dataset expects the following directory structure:

    <root>/
    └── <speaker>/
        └── <vowel (IPA)>/
            └── <modality>/
                └── 0001.flac
                └── 0002.flac
                ...

Example:
    Recordings/
    └── Hannah/
        ├── ɔ/
        │   └── Sung_M2/
        │       ├── 0001.flac
        │       └── 0002.flac
        └── ɐː/
            └── Spoken/
                └── 0001.flac

The vowel folder name is used directly as the IPA label — use UTF-8
characters in folder names (e.g. "ɐː" not "a:").

Formants are estimated from audio via Praat (parselmouth). The estimation
parameters are tuned for speech; sung samples at ~200Hz F0 will give
reliable F1–F3 but F4 should be treated with caution due to sparse
harmonics at that pitch.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import librosa


# ---------------------------------------------------------------------------
# Audio file I/O
# ---------------------------------------------------------------------------

def load_audio_file(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load a single audio file, resampling to `sr` if needed.

    Returns
    -------
    (audio, sample_rate)
    """
    audio, sr = librosa.load(path, sr=sr)
    return audio, sr


def load_audio_files_from_folder(
    folder: str, ext: str = ".wav", sr: int = 16000
) -> list[tuple[str, np.ndarray]]:
    """
    Load all audio files with a given extension from a folder.

    Returns
    -------
    list of (filepath, audio_array) tuples, sorted by filename
    """
    items = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(ext):
            path = os.path.join(folder, fname)
            audio, _ = load_audio_file(path, sr=sr)
            items.append((path, audio))
    return items


def prepare_labeled_dataset(
    file_label_pairs, sr: int = 16000
) -> list[dict]:
    """
    Build a dataset from (filepath_or_array, label) pairs.

    Parameters
    ----------
    file_label_pairs : iterable of (str | np.ndarray, label)
        If the first element is a string it is treated as a file path
        and loaded; otherwise it is used directly as an audio array.

    Returns
    -------
    list of dicts with keys "audio", "label", and optionally "path"
    """
    dataset = []
    for item, label in file_label_pairs:
        if isinstance(item, str):
            audio, _ = load_audio_file(item, sr=sr)
            dataset.append({"audio": audio, "label": label, "path": item})
        else:
            dataset.append({"audio": item, "label": label})
    return dataset


# ---------------------------------------------------------------------------
# Vowel nucleus extraction
# ---------------------------------------------------------------------------

def extract_vowel_nucleus(
    audio: np.ndarray, sr: int = 16000, center_ms: int = 100
) -> np.ndarray:
    """
    Extract a fixed-length window centred on the middle of the signal.

    This is a simple heuristic for isolating the vowel nucleus; for
    carefully segmented recordings it is usually sufficient.

    Parameters
    ----------
    audio     : 1-D float array
    sr        : sample rate in Hz
    center_ms : window duration in milliseconds

    Returns
    -------
    1-D float array of length <= center_ms * sr / 1000
    """
    center = len(audio) // 2
    half   = int(sr * center_ms / 1000 / 2)
    return audio[max(0, center - half) : min(len(audio), center + half)]


# ---------------------------------------------------------------------------
# Formant estimation via Praat
# ---------------------------------------------------------------------------

def estimate_formants(
    audio: np.ndarray,
    sr: int = 16000,
    n_formants: int = 4,
    max_formant_hz: float = 5500.0,
    frame_shift_ms: float = 10.0,
    window_ms: float = 25.0,
    min_f0_hz: float | None = None,
) -> np.ndarray:
    """
    Estimate per-frame formants using Praat via parselmouth.

    For sung audio at high F0 (>150Hz), F1 tracking is unreliable because
    the vocal tract resonance gets pulled toward the nearest harmonic.
    Set ``min_f0_hz`` to the approximate lowest F0 in the recording — this
    widens the analysis window to at least 3 pitch periods and raises Praat's
    internal formant floor, giving more reliable F2/F3 estimates even when
    F1 is locked to a harmonic.

    Rule of thumb: at F0=200Hz, only F2 and F3 are reliably trackable.
    Pass ``use_formants=(1, 2)`` in ``speaker_config`` for such samples.

    Parameters
    ----------
    audio          : 1-D float32 audio array
    sr             : sample rate in Hz
    n_formants     : number of formants to extract (default 4)
    max_formant_hz : frequency ceiling passed to Praat's Burg formant tracker.
                     5500 Hz is standard for adult female speech;
                     use 5000 for adult male, 6000-8000 for children.
    frame_shift_ms : analysis frame shift in milliseconds
    window_ms      : analysis window length in milliseconds. When min_f0_hz
                     is set, this is automatically raised to cover at least
                     3 pitch periods if the default would be too short.
    min_f0_hz      : lowest expected F0 in Hz. When provided:
                     - window_ms is extended to max(window_ms, 3000/min_f0_hz)
                     - Praat's formant floor is set to min_f0_hz/2, preventing
                       F1 being reported below the fundamental

    Returns
    -------
    (T, n_formants) float32 array of formant frequencies in Hz.
    Frames where Praat could not estimate a formant are filled with NaN
    and then interpolated so downstream code sees no gaps.
    """
    try:
        import parselmouth
    except ImportError:
        raise ImportError(
            "parselmouth is required for formant estimation. "
            "Install it with: pip install praat-parselmouth"
        )

    # Widen window to cover at least 3 pitch periods for high-F0 audio
    effective_window_ms = window_ms
    if min_f0_hz is not None:
        min_window_ms = 3000.0 / min_f0_hz   # 3 periods in ms
        effective_window_ms = max(window_ms, min_window_ms)

    snd     = parselmouth.Sound(audio, sampling_frequency=sr)
    formant = snd.to_formant_burg(
        time_step              = frame_shift_ms / 1000.0,
        max_number_of_formants = n_formants,
        maximum_formant        = max_formant_hz,
        window_length          = effective_window_ms / 1000.0,
        pre_emphasis_from      = 50.0,
    )

    times  = formant.ts()
    frames = np.full((len(times), n_formants), np.nan, dtype=np.float32)
    for t_idx, t in enumerate(times):
        for f_idx in range(n_formants):
            val = formant.get_value_at_time(f_idx + 1, t)
            if val is not None and not np.isnan(val):
                frames[t_idx, f_idx] = val

    # Interpolate NaN frames (occur at signal edges or tracking failures)
    for f_idx in range(n_formants):
        col  = frames[:, f_idx]
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            idx = np.where(~mask)[0]
            frames[:, f_idx] = np.interp(np.arange(len(col)), idx, col[idx])

    # If min_f0_hz set, clamp any F1 values below the fundamental — these
    # are tracking failures where Praat locked onto a subharmonic
    if min_f0_hz is not None:
        frames[:, 0] = np.maximum(frames[:, 0], min_f0_hz)

    return frames


# ---------------------------------------------------------------------------
# Personal dataset loader
# ---------------------------------------------------------------------------

# Default max_formant_hz per speaker group — can be overridden per speaker
# in the speakers config passed to load_personal_dataset.
_DEFAULT_MAX_FORMANT = 5500.0  # conservative default; works for most adult speech



def load_personal_dataset(
    root: str,
    sr: int = 16000,
    extensions: tuple[str, ...] = (".flac", ".wav", ".mp3"),
    speaker_config: "dict[str, dict] | None" = None,
    dialect: str = "unknown",
    source: str | None = None,
) -> list[dict]:
    """
    Load a personal vowel recording dataset from a directory tree.

    Expected structure::

        <root>/
        <speaker>/
            <vowel>/
                <modality>/
                    *.flac

    The vowel folder name is used as the IPA label directly.

    Parameters
    ----------
    root           : path to the root recordings directory
    sr             : target sample rate for audio loading
    extensions     : audio file extensions to include
    speaker_config : optional dict mapping speaker name to kwargs for
                     estimate_formants(), e.g.
                     {"Hannah": {"max_formant_hz": 5500}}
    dialect        : BCP-47-style dialect tag for all samples in this
                     dataset, e.g. "en-AU", "en-GB-RP". Default "unknown".
    source         : short dataset identifier. Defaults to root folder name.

    Returns
    -------
    list of sample dicts with full pipeline schema:
        audio        : (T,) float32
        label        : str  -- IPA vowel label from folder name
        group        : str  -- speaker name
        speaker      : str  -- speaker name
        dialect      : str  -- BCP-47 dialect tag
        modality     : str  -- modality folder name, e.g. "Sung_M2"
        source       : str  -- dataset identifier
        formants     : (4,) float32 -- mean F1-F4 in Hz
        formants_std : (4,) float32 -- std  F1-F4 in Hz
        formants_raw : (T, 4) float32 -- per-frame F1-F4 in Hz
        path         : str  -- source file path
    """
    root   = Path(root)
    source = source or root.name
    if not root.is_dir():
        raise ValueError(f"Root directory not found: {root}")

    speaker_config   = speaker_config or {}
    dataset          = []
    extensions_lower = tuple(e.lower() for e in extensions)

    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name

        for vowel_dir in sorted(speaker_dir.iterdir()):
            if not vowel_dir.is_dir():
                continue
            vowel = vowel_dir.name

            for modality_dir in sorted(vowel_dir.iterdir()):
                if not modality_dir.is_dir():
                    continue
                modality = modality_dir.name

                audio_files    = sorted(
                    p for p in modality_dir.iterdir()
                    if p.suffix.lower() in extensions_lower
                )
                formant_kwargs = speaker_config.get(speaker, {})

                for audio_path in audio_files:
                    try:
                        audio, _ = librosa.load(
                            str(audio_path), sr=sr, mono=True, dtype=np.float32
                        )
                        peak = np.abs(audio).max()
                        if peak > 1e-6:
                            audio = audio / peak

                        formants_raw = estimate_formants(
                            audio, sr=sr, **formant_kwargs
                        )

                        dataset.append({
                            "audio":        audio,
                            "label":        vowel,
                            "group":        speaker,
                            "speaker":      speaker,
                            "dialect":      dialect,
                            "modality":     modality,
                            "source":       source,
                            "formants":     formants_raw.mean(axis=0),
                            "formants_std": formants_raw.std(axis=0),
                            "formants_raw": formants_raw,
                            "path":         str(audio_path),
                        })
                    except Exception as exc:
                        print(f"  WARNING: skipping {audio_path}: {exc}")

    if not dataset:
        raise ValueError(
            f"No audio files found under {root}. "
            f"Expected structure: <root>/<speaker>/<vowel>/<modality>/*.flac"
        )

    print(
        f"Loaded {len(dataset)} samples from {root} "
        f"({len({d['speaker'] for d in dataset})} speakers, "
        f"{len({d['label'] for d in dataset})} vowel classes, "
        f"dialect={dialect!r})"
    )
    return dataset
