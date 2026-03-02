# datasets_hf.py
"""
HuggingFace dataset loaders.

Each loader returns a list of dicts compatible with the embedding pipeline:

    {
        "audio"         : np.ndarray,   # (T,) float32
        "label"         : str,          # IPA vowel label
        "group"         : str,          # demographic group (men/women/boys/girls/...)
        "speaker"       : str,          # unique speaker ID within dataset
        "dialect"       : str,          # BCP-47-style dialect tag
        "modality"      : str,          # "spoken", "sung", etc.
        "source"        : str,          # dataset identifier
        "formants"      : np.ndarray,   # (4,) mean F1-F4 in Hz  [mu]
        "formants_std"  : np.ndarray,   # (4,) std  F1-F4 in Hz  [sigma]
        "formants_raw"  : np.ndarray,   # (T, 4) per-frame F1-F4 in Hz
    }

Schema notes
------------
label
    UTF-8 IPA characters, faithful to each dataset's own phonological
    categories. Cross-dataset normalisation (e.g. collapsing rhotic /er/
    with non-rhotic /ew/) should happen at analysis time, not load time.

group
    Coarse demographic grouping. Used by the VTL group-stats reporter and
    for cross-dataset stratification. Hillenbrand uses "men"/"women"/
    "boys"/"girls"; personal datasets use the speaker name as group if no
    finer demographic mapping is provided.

speaker
    Unique ID within the dataset. For Hillenbrand this is parsed from the
    filename (e.g. "m01", "w14"). For personal datasets it is the speaker
    folder name. Intended for per-speaker VTL consistency analysis and
    future speaker-identity clustering.

dialect
    BCP-47-style tag, e.g. "en-US-GenAm", "en-AU". Kept intentionally
    coarse -- sub-dialect distinctions should be captured in speaker-level
    metadata rather than this field.

modality
    Recording modality: "spoken" for read-speech datasets, "sung" (with
    optional pitch annotation, e.g. "sung-M2") for sung samples.

source
    Short dataset identifier, e.g. "hillenbrand1995". Useful for filtering
    and colour-coding in cross-dataset visualisations.

Vowel label mappings
--------------------
Hillenbrand et al. (1995) hVd frame dataset -- General American English.
Source: Hillenbrand et al. (1995), JASA 97(5), Table I.
/e/ and /o/ are phonological monophthongs per the original transcription.
/er/ is the stressed r-coloured mid-central vowel (rhotic) -- distinct from
non-rhotic /ew/ by both phonological category and acoustic realisation (F3).
"""

from __future__ import annotations

import ast
import re

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Vowel label mappings
# ---------------------------------------------------------------------------

HILLENBRAND_TO_IPA: dict[str, str] = {
    "ae": "æ",   # had
    "ah": "ɑ",   # hod
    "aw": "ɔ",   # hawed
    "eh": "ɛ",   # head
    "ei": "e",   # haid   (/e/ monophthong per original transcription)
    "er": "ɝ",   # heard  (stressed rhotic -- distinct from non-rhotic /ɜː/)
    "ih": "ɪ",   # hid
    "iy": "i",   # heed
    "oa": "o",   # boat   (/o/ monophthong per original transcription)
    "oo": "ʊ",   # hood
    "uh": "ʌ",   # hud
    "uw": "u",   # who'd
}

# Hillenbrand group codes in filenames -> demographic label
_HILLENBRAND_GROUP_CODE: dict[str, str] = {
    "m": "men",
    "w": "women",
    "b": "boys",
    "g": "girls",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_hillenbrand_speaker(item: dict, first_item: bool = False) -> str:
    """
    Extract a speaker ID from a Hillenbrand dataset item.

    The original filenames encode speaker identity as:
        <group_char><talker_number><vowel_code>.wav
        e.g. m01ae.wav -> man 01, vowel /ae/

    We try several candidate column names in order of likelihood.
    Falls back to "unknown" if no filename column is found, printing
    a one-time warning so the caller knows speaker ID is unavailable.

    Returns e.g. "m01", "w14", "b03", "g07".
    """
    for col in ("filename", "file", "file_name", "id", "name"):
        val = item.get(col)
        if val and isinstance(val, str):
            m = re.match(r'^([mwbg])(\d{2})', val.lower())
            if m:
                return m.group(1) + m.group(2)

    if first_item:
        available = [k for k, v in item.items()
                     if isinstance(v, str) and len(v) < 50]
        print(
            f"\nWARNING: Could not find a filename column to parse speaker ID.\n"
            f"  Available short string columns: {available}\n"
            f"  Speaker IDs will be set to 'unknown'.\n"
            f"  If a filename column is listed above, please open an issue.\n"
        )
    return "unknown"


def _hillenbrand_group_from_speaker(speaker: str) -> str:
    """Map speaker ID like 'm01' to demographic group label."""
    if speaker == "unknown":
        return "unknown"
    code = speaker[0].lower()
    return _HILLENBRAND_GROUP_CODE.get(code, "unknown")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_hillenbrand(split: str = "train", sr: int = 16000) -> list[dict]:
    """
    Load the MLSpeech/hillenbrand_vowels dataset from HuggingFace.

    Formant columns contain string-encoded lists of ~10ms-interval frame
    measurements across the vowel. All three representations (raw frames,
    mu, sigma) are returned so callers can choose what they need.

    Fields added beyond the raw dataset
    ------------------------------------
    label    : IPA symbol via HILLENBRAND_TO_IPA
    speaker  : parsed from filename column (e.g. "m01") -- falls back to
               "unknown" with a warning if no filename column is found
    group    : derived from speaker code ("men"/"women"/"boys"/"girls"),
               falling back to the raw "group" column if parsing fails
    dialect  : "en-US-GenAm" (all Hillenbrand speakers are General American)
    modality : "spoken" (all Hillenbrand recordings are read speech)
    source   : "hillenbrand1995"

    Parameters
    ----------
    split : dataset split to load (default "train")
    sr    : target sample rate (for schema consistency; audio is already 16kHz)

    Returns
    -------
    list of sample dicts -- see module docstring for full schema
    """
    ds = load_dataset("MLSpeech/hillenbrand_vowels", split=split, streaming=False)

    dataset = []
    for idx, item in enumerate(tqdm(ds, desc="Loading Hillenbrand")):
        audio = np.array(ast.literal_eval(item["audio"]), dtype=np.float32)

        frame_arrays = [
            np.array(ast.literal_eval(item[f"formant_{i}"]), dtype=np.float32)
            for i in range(1, 5)
        ]
        min_frames   = min(len(f) for f in frame_arrays)
        formants_raw = np.stack([f[:min_frames] for f in frame_arrays], axis=1)

        raw_label = item["vowel"]
        ipa_label = HILLENBRAND_TO_IPA.get(raw_label, raw_label)

        speaker = _parse_hillenbrand_speaker(item, first_item=(idx == 0))
        group   = _hillenbrand_group_from_speaker(speaker)
        if group == "unknown":
            group = item.get("group", "unknown")

        dataset.append({
            "audio":        audio,
            "label":        ipa_label,
            "group":        group,
            "speaker":      speaker,
            "dialect":      "en-US-GenAm",
            "modality":     "spoken",
            "source":       "hillenbrand1995",
            "formants":     formants_raw.mean(axis=0),
            "formants_std": formants_raw.std(axis=0),
            "formants_raw": formants_raw,
        })

    n_speakers = len({d["speaker"] for d in dataset})
    print(f"Loaded {len(dataset)} Hillenbrand samples "
          f"({n_speakers} speakers, {len({d['label'] for d in dataset})} vowel classes)")
    return dataset
