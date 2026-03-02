# whisper.py
"""
Whisper encoder loading and hidden-state extraction.

Keeps all HuggingFace / PyTorch concerns isolated from caching and
dataset logic.
"""

from __future__ import annotations

import torch
import numpy as np
from transformers import WhisperProcessor, WhisperModel


class WhisperEncoder:
    """
    Thin wrapper around a frozen Whisper encoder.

    Parameters
    ----------
    model_name : HuggingFace model identifier, e.g. "openai/whisper-small"
    device     : "cuda", "cpu", or None (auto-detected)
    """

    def __init__(self, model_name: str = "openai/whisper-small", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name} on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def num_layers(self) -> int:
        """Total number of hidden states (including embedding layer 0)."""
        return self.model.config.encoder_layers + 1

    def extract_all_layers(self, audio: np.ndarray, sample_rate: int = 16000) -> list[np.ndarray]:
        """
        Run the encoder and return mean-pooled embeddings for every layer.

        Parameters
        ----------
        audio       : 1-D float32 array
        sample_rate : audio sample rate in Hz

        Returns
        -------
        list of (hidden_dim,) arrays, one per encoder layer (including layer 0)
        """
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_features, output_hidden_states=True
            )

        # hidden_states is a tuple of (batch=1, time, dim) tensors
        return [
            h.squeeze(0).cpu().numpy()  # (time, dim) — pooling done by caller
            for h in encoder_outputs.hidden_states
        ]
