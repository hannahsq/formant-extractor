# heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeHeadPooled(nn.Module):
    def __init__(self, d_phys, d_vowel, num_classes, formant_dim=4):
        super().__init__()
        self.formant_head = FormantHead(d_phys, formant_dim, hidden=1024)
        self.vowel_head   = VowelHead(d_vowel, num_classes, hidden=768)
        self.vtl_head     = nn.Linear(d_phys, 1)  # simple linear head

    def forward(self, x_phys, x_vowel):
        return {
            "formants": self.formant_head(x_phys),   # whitened formants
            "vowels":   self.vowel_head(x_vowel),
            "vtl":      self.vtl_head(x_phys).squeeze(-1),  # scalar
        }


class PhysHead(nn.Module):
    def __init__(self, in_dim, out_dim=5, hidden=512, dropout=0.1):
        super().__init__()
        self.resblock = ResidualMLP(in_dim, hidden, dropout)
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.resblock(x)
        return self.fc_out(h)

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        # First layer
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)

        # Second layer
        h2 = F.gelu(self.fc2(h))
        h2 = self.dropout(h2)

        # Residual connection + layer norm
        return self.norm(h + h2)


class VowelHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=512, dropout=0.1):
        super().__init__()
        self.resblock = ResidualMLP(in_dim, hidden, dropout)
        self.fc_out = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.resblock(x)
        return self.fc_out(h)


class FormantHead(nn.Module):
    def __init__(self, in_dim, out_dim=4, hidden=512, dropout=0.1):
        super().__init__()
        self.resblock = ResidualMLP(in_dim, hidden, dropout)
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.resblock(x)
        return self.fc_out(h)

# class TwoHeadPooledPhys(nn.Module):
#     def __init__(self, d_phys, d_vowel, num_classes):
#         super().__init__()
#         self.phys_head  = PhysHead(d_phys, out_dim=5, hidden=2048)
#         self.vowel_head = VowelHead(d_vowel, num_classes)

#     def forward(self, x_phys, x_vowel):
#         return {
#             "phys":   self.phys_head(x_phys),
#             "vowels": self.vowel_head(x_vowel),
#         }
    
class TwoHeadPooledPhys(nn.Module):
    def __init__(self, d_phys, d_vowel, num_classes):
        super().__init__()
        self.phys_head  = PhysHead(d_phys, out_dim=5, hidden=2048)
        self.vowel_head = VowelHead(d_vowel + 4, num_classes)

    def forward(self, x_phys, x_vowel):
        form_pred = self.phys_head(x_phys)
        vowel_in = torch.cat([x_vowel, form_pred[:, 1:]], dim=-1)
        return {
            "phys":   form_pred,
            "vowels": self.vowel_head(vowel_in),
        }



# class VowelHead(nn.Module):
#     def __init__(self, in_dim, num_classes, hidden=512, dropout=0.1):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hidden)
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(hidden, num_classes)

#     def forward(self, x):
#         x = F.gelu(self.fc1(x))
#         x = self.dropout(x)
#         return self.fc2(x)


# class FormantHead(nn.Module):
#     def __init__(self, in_dim, out_dim=4, hidden=512, dropout=0.1):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hidden)
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(hidden, out_dim)

#     def forward(self, x):
#         x = F.gelu(self.fc1(x))
#         x = self.dropout(x)
#         return self.fc2(x)

class TwoHeadPooled(nn.Module):
    def __init__(self, d_model, num_classes, formant_dim=4):
        super().__init__()
        self.formant_head = FormantHead(d_model, formant_dim)
        self.vowel_head   = VowelHead(d_model, num_classes)

    def forward(self, pooled_l4, pooled_l12):
        return {
            "formants": self.formant_head(pooled_l4),
            "vowels":   self.vowel_head(pooled_l12),
        }

class WhisperTwoHead(nn.Module):
    def __init__(self, d_model, num_classes, formant_dim=4):
        super().__init__()
        self.formant_head = FormantHead(d_model, formant_dim)
        self.vowel_head   = VowelHead(d_model, num_classes)

    def forward(self, hidden_states):
        # hidden_states is a list: one tensor per layer
        h4  = hidden_states[4].mean(dim=1)   # (B, D)
        h12 = hidden_states[12].mean(dim=1)

        return {
            "formants": self.formant_head(h4),
            "vowels":   self.vowel_head(h12),
        }
