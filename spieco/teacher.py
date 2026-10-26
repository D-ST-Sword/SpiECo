"""Complete-input teacher used for initialization and training-only targets."""
import torch
from torch import nn
from .layers import ModalityEncoder, TextAMP, UnalignedPair
from .spiking import TemporalAdapter

DIMS = {"urfunny": (1024, 81, 371), "mosi": (1024, 5, 20), "sims": (768, 33, 709)}
MODALITIES = ("text", "audio", "vision")
PAIRS = {"TA": ("text", "audio"), "TV": ("text", "vision"), "AV": ("audio", "vision")}


class TextEncoder(nn.Module):
    def __init__(self, path, hidden_size):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(path, local_files_only=True)
        if self.model.config.hidden_size != hidden_size:
            raise ValueError(f"Expected text width {hidden_size}; check the dataset/encoder pairing")

    def forward(self, tokens):
        ids = tokens[:, 0].long().clamp(0, self.model.config.vocab_size - 1)
        return self.model(input_ids=ids, attention_mask=tokens[:, 1].float()).last_hidden_state


class TeacherCore(nn.Module):
    def __init__(self, dims, encoder):
        super().__init__()
        self.pretrained_text_encoder = TextAMP(encoder)
        self.encoders = nn.ModuleDict({m: ModalityEncoder(d, 128, .1, m) for m, d in zip(MODALITIES, dims)})
        self.pairs = nn.ModuleDict({name: UnalignedPair() for name in PAIRS})
        self.trimodal = nn.Sequential(nn.Linear(384, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.1))
        # Frozen compatibility slots preserve archived state keys and initialization order.
        self.heads = nn.ModuleDict({name: nn.Linear(128, 1) for name in ("T", "A", "V", "TA", "TV", "AV", "TAV")})
        for name, head in self.heads.items():
            head.requires_grad_(name == "TAV")


class Teacher(nn.Module):
    def __init__(self, dataset, encoder_path=None, encoder=None):
        super().__init__()
        dims = DIMS[dataset]
        if encoder is None:
            if encoder_path is None:
                raise ValueError("Training requires a local pretrained text encoder")
            encoder = TextEncoder(encoder_path, dims[0])
        self.core = TeacherCore(dims, encoder)
        self.classes = 1
        self.adapters = nn.ModuleDict({m: TemporalAdapter() for m in ("audio", "vision")})

    def forward(self, batch):
        masks = {m: batch["text"][:, 1].bool() if m == "text" else batch[m + "_mask"] for m in MODALITIES}
        states = {}
        for m in MODALITIES:
            value = self.core.pretrained_text_encoder(batch[m]) if m == "text" else batch[m]
            value = self.core.encoders[m](value, masks[m])
            states[m] = self.adapters[m](value, masks[m]) if m in self.adapters else value
        pairs = [self.core.pairs[n](states[a], states[b], masks[a], masks[b]) for n, (a, b) in PAIRS.items()]
        latent = self.core.trimodal(torch.cat(pairs, -1))
        return {"M": self.core.heads["TAV"](latent)[:, 0], "latent": latent}
