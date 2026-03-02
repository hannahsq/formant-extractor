# dataset.py
import os
import librosa
import numpy as np

def load_audio_file(path, sr=16000):
    audio, sr = librosa.load(path, sr=sr)
    return audio, sr

def load_audio_files_from_folder(folder, ext=".wav", sr=16000):
    items = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(ext):
            path = os.path.join(folder, fname)
            audio, _ = load_audio_file(path, sr=sr)
            items.append((path, audio))
    return items

def prepare_labeled_dataset(file_label_pairs, sr=16000):
    """
    file_label_pairs: list of (filepath, label) or (audio_array, label)
    returns: list of dicts { 'audio': array, 'label': label, 'path': optional }
    """
    dataset = []
    for item, label in file_label_pairs:
        if isinstance(item, str):
            audio, _ = load_audio_file(item, sr=sr)
            dataset.append({'audio': audio, 'label': label, 'path': item})
        else:
            dataset.append({'audio': item, 'label': label})
    return dataset

def extract_vowel_nucleus(audio, sr=16000, center_ms=100):
    """
    Simple heuristic: take center window of length center_ms
    """
    center = len(audio) // 2
    half = int(sr * center_ms / 1000 / 2)
    start = max(0, center - half)
    end = min(len(audio), center + half)
    return audio[start:end]
