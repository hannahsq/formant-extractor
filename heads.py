# heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """Two-layer MLP with residual connection, GELU activations, and LayerNorm."""

    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(in_dim, hidden)
        self.fc2     = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h  = self.dropout(F.gelu(self.fc1(x)))
        h2 = self.dropout(F.gelu(self.fc2(h)))
        return self.norm(h + h2)


class VowelHead(nn.Module):
    """Classification head for vowel identity."""

    def __init__(self, in_dim: int, num_classes: int,
                 hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.resblock = ResidualMLP(in_dim, hidden, dropout)
        self.fc_out   = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.resblock(x))


class FormantHead(nn.Module):
    """Regression head for whitened formant prediction."""

    def __init__(self, in_dim: int, out_dim: int = 4,
                 hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.resblock = ResidualMLP(in_dim, hidden, dropout)
        self.fc_out   = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.resblock(x))


class PhysHead(nn.Module):
    """
    Combined physical head predicting [invL_norm, F1_white, F2_white, F3_white, F4_white].
    Used by TwoHeadPooledPhys (blended-VTL pipeline).
    """

    def __init__(self, in_dim: int, out_dim: int = 5,
                 hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.resblock = ResidualMLP(in_dim, hidden, dropout)
        self.fc_out   = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.resblock(x))


class ThreeHeadPooled(nn.Module):
    """
    Three-head model operating on pre-pooled layer embeddings.

    Heads
    -----
    formants : FormantHead — predicts whitened F1–F4 (sample-mode VTL pipeline)
    vowels   : VowelHead   — predicts vowel class logits
    vtl      : Linear      — predicts normalised inverse-VTL scalar

    Parameters
    ----------
    d_phys      : input dimension for formant + VTL heads (concatenated phys layers)
    d_vowel     : input dimension for vowel head (concatenated vowel layers)
    num_classes : number of vowel classes
    formant_dim : number of formants to predict (default 4)
    """

    def __init__(self, d_phys: int, d_vowel: int,
                 num_classes: int, formant_dim: int = 4):
        super().__init__()
        self.d_phys      = d_phys
        self.d_vowel     = d_vowel
        self.num_classes = num_classes
        self.formant_dim = formant_dim
        self.formant_head = FormantHead(d_phys, formant_dim, hidden=1024)
        self.vowel_head   = VowelHead(d_vowel, num_classes, hidden=768)
        self.vtl_head     = nn.Linear(d_phys, 1)

    def forward(self, x_phys: torch.Tensor,
                x_vowel: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "formants": self.formant_head(x_phys),
            "vowels":   self.vowel_head(x_vowel),
            "vtl":      self.vtl_head(x_phys).squeeze(-1),
        }


class TwoHeadPooledPhys(nn.Module):
    """
    Two-head model using the blended-VTL physical pipeline.

    The PhysHead predicts a 5-vector [invL_norm, F1_white, ..., F4_white].
    The predicted normalised formants (indices 1–4) are concatenated onto the
    vowel embeddings before classification, allowing the vowel head to condition
    on physical structure.

    Parameters
    ----------
    d_phys      : input dimension for physical head
    d_vowel     : input dimension for vowel head (before formant concat)
    num_classes : number of vowel classes
    """

    def __init__(self, d_phys: int, d_vowel: int, num_classes: int):
        super().__init__()
        self.d_phys      = d_phys
        self.d_vowel     = d_vowel
        self.num_classes = num_classes
        self.phys_head  = PhysHead(d_phys, out_dim=5, hidden=2048)
        self.vowel_head = VowelHead(d_vowel + 4, num_classes)

    def forward(self, x_phys: torch.Tensor,
                x_vowel: torch.Tensor) -> dict[str, torch.Tensor]:
        phys_pred = self.phys_head(x_phys)
        vowel_in  = torch.cat([x_vowel, phys_pred[:, 1:]], dim=-1)
        return {
            "phys":   phys_pred,
            "vowels": self.vowel_head(vowel_in),
        }
