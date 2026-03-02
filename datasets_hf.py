# # datasets_hf.py
# from datasets import load_dataset
# import numpy as np

# def load_hillenbrand(split="train"):
#     """
#     Loads the MLSpeech/hillenbrand_vowels dataset from Hugging Face.
#     Returns a list of dicts: { 'audio': np.array, 'label': vowel_string }
#     """
#     ds = load_dataset("MLSpeech/hillenbrand_vowels", split=split)
#     dataset = []
#     for item in ds:
#         # Hugging Face audio is already decoded to a dict with 'array' and 'sampling_rate'
#         audio = eval(item["audio"])
#         label = item["vowel"]  # e.g., "iy", "ae", "uw", etc.

#         dataset.append({
#             "audio": np.array(audio, dtype=np.float32),
#             "label": label
#         })

#     return dataset

# datasets_hf.py
from datasets import load_dataset, Audio
import numpy as np
from tqdm import tqdm
import ast

def parse_audio_string(s):
    return np.array(ast.literal_eval(s), dtype=np.float32)

# def load_hillenbrand(split="train", sr=16000):
#     """
#     Load the MLSpeech/hillenbrand_vowels dataset with audio decoding enabled.
#     """
#     ds = load_dataset(
#         "MLSpeech/hillenbrand_vowels",
#         split=split,
#         streaming=False
#     ).select(range(200))

#     # Force audio decoding
#     #ds = ds.map(eval_string_column).cast_column("audio_array", Audio(sampling_rate=sr))
    
#     dataset = []
#     for item in tqdm(ds):
#         #audio = item["audio"]["array"]
#         audio = parse_audio_string(item["audio"])
#         label = item["vowel"]

#         dataset.append({
#             "audio": audio,
#             "label": label
#         })

#     return dataset

import ast

def load_hillenbrand(split="train", sr=16000):
    ds = load_dataset("MLSpeech/hillenbrand_vowels", split=split, streaming=False)#.select(range(200))

    dataset = []
    for item in ds:
        audio = np.array(ast.literal_eval(item["audio"]), dtype=np.float32)

        # Formants are stored as strings like "[F1, F2, F3, F4]"
        formants = np.array([ast.literal_eval(item["formant_1"]),ast.literal_eval(item["formant_2"]),ast.literal_eval(item["formant_3"]),ast.literal_eval(item["formant_4"])])
        formants = [np.median(f) for f in formants]

        dataset.append({
            "audio": audio,
            "label": item["vowel"],
            "group": item["group"],
            "formants": formants
        })

    return dataset
