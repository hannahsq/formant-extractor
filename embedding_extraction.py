# embedding_extraction.py
import torch
from transformers import WhisperProcessor, WhisperModel

def load_model(model_name="openai/whisper-small", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor, device

def extract_layer_embeddings(model, processor, audio_array, sample_rate, layer_index=None, device=None):
    """
    audio_array: 1D numpy or torch array
    layer_index: None -> return last_hidden_state; int -> return that layer
    returns: tensor shape (time_steps, hidden_dim)
    """
    device = device or next(model.parameters()).device
    # preprocess audio to model input features
    inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    # run encoder and collect hidden states
    with torch.no_grad():
        encoder_outputs = model.encoder(input_features, output_hidden_states=True)
    hidden_states = encoder_outputs.hidden_states  # tuple: (layer0, layer1, ..., last)
    if layer_index is None:
        return hidden_states[-1].squeeze(0).cpu()
    else:
        return hidden_states[layer_index].squeeze(0).cpu()

def extract_all_layers(model, processor, audio_array, sample_rate, device=None):
    device = device or next(model.parameters()).device
    inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        encoder_outputs = model.encoder(input_features, output_hidden_states=True)
    hidden_states = [h.squeeze(0).cpu() for h in encoder_outputs.hidden_states]
    return hidden_states  # list of tensors per layer