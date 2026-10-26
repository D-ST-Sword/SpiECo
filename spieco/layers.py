"""Private encoders and unaligned bidirectional interaction used by SpiECo."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def mask_states(x, mask):
    return x.masked_fill(~mask.unsqueeze(-1), 0)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, maximum=2048):
        super().__init__()
        position = torch.arange(maximum, dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        values = torch.zeros(maximum, dim)
        values[:, 0::2], values[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        self.register_buffer("values", values, persistent=False)

    def forward(self, x):
        if x.size(1) > len(self.values):
            raise ValueError("Dialogue exceeds positional encoding capacity")
        return x + self.values[:x.size(1)]


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, dim, dropout, kind):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.kind = kind
        if kind == "text":
            self.positions = PositionalEncoding(dim)
            layer = nn.TransformerEncoderLayer(dim, 4, 2 * dim, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        else:
            self.context = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        x = mask_states(self.projection(x), mask)
        if self.kind == "text":
            output = self.context(self.positions(x), src_key_padding_mask=~mask)
        else:
            packed = pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
            output, _ = self.context(packed)
            output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.size(1))
        return mask_states(self.output_norm(output + x), mask)


def pool(state, mask):
    return (state * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class TextAMP(nn.Module):
    def __init__(self, encoder, amp=True):
        super().__init__()
        self.encoder, self.amp = encoder, amp
        self.encoder.model.gradient_checkpointing_enable()

    def forward(self, tokens):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and tokens.is_cuda):
            return self.encoder(tokens).float()


class UnalignedPair(nn.Module):
    def __init__(self, dim=128, dropout=.1):
        super().__init__()
        self.left_norm, self.right_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.forward_attention = nn.MultiheadAttention(dim, 8, dropout=dropout, batch_first=True)
        self.reverse_attention = nn.MultiheadAttention(dim, 8, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(2 * dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.norm, self.ffnorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, left, right, left_mask, right_mask):
        a, b = self.left_norm(left), self.right_norm(right)
        x = pool(self.forward_attention(a, b, b, key_padding_mask=~right_mask, need_weights=False)[0], left_mask)
        y = pool(self.reverse_attention(b, a, a, key_padding_mask=~left_mask, need_weights=False)[0], right_mask)
        gate = torch.sigmoid(self.gate(torch.cat([x, y], -1)))
        residual = .5 * (pool(left, left_mask) + pool(right, right_mask))
        state = self.norm(residual + self.dropout(gate * x + (1 - gate) * y))
        return state + self.dropout(self.ff(self.ffnorm(state)))
