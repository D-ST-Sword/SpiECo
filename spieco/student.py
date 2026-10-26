"""Training-only consolidation and independently executable audiovisual student."""
import copy
import torch
from torch import nn
from .layers import ModalityEncoder, UnalignedPair, pool
from .spiking import TemporalAdapter as SequenceAdapter

DIMS = {"urfunny": (81, 371), "mosi": (5, 20), "sims": (33, 709)}


class TemporalAdapter(SequenceAdapter):
    def forward(self, x, mask=None):
        if x.ndim == 3 and x.shape[1] == 1:
            return super().forward(x[:, 0])[:, None].masked_fill(~mask[..., None], 0)
        return super().forward(x, mask)


def bridge():
    return nn.Sequential(nn.Linear(384, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.1))


class AVOnly(nn.Module):
    def __init__(self, dataset, teacher=None):
        super().__init__()
        self.classes = 1
        self.encoders = nn.ModuleDict({m: ModalityEncoder(size, 128, .1, m)
            for m, size in zip(("audio", "vision"), DIMS[dataset])})
        self.adapters = nn.ModuleDict({m: TemporalAdapter() for m in ("audio", "vision")})
        self.pair = UnalignedPair()
        self.bridge = bridge()
        self.head = nn.Linear(128, self.classes)
        if teacher is not None:
            for m in ("audio", "vision"):
                self.encoders[m].load_state_dict(teacher.core.encoders[m].state_dict(), strict=True)
                self.adapters[m].load_state_dict(teacher.adapters[m].state_dict(), strict=True)
            self.pair.load_state_dict(teacher.core.pairs["AV"].state_dict(), strict=True)
            self.bridge.load_state_dict(teacher.core.trimodal.state_dict(), strict=True)
            self.head.load_state_dict(teacher.core.heads["TAV"].state_dict(), strict=True)

    def encode(self, batch):
        states = {m: self.adapters[m](self.encoders[m](batch[m], batch[m + "_mask"]), batch[m + "_mask"])
                  for m in ("audio", "vision")}
        av = self.pair(states["audio"], states["vision"], batch["audio_mask"], batch["vision_mask"])
        features = torch.cat([pool(states["audio"], batch["audio_mask"]),
                              pool(states["vision"], batch["vision_mask"]), av], -1)
        latent = self.bridge(features)
        prediction = self.head(latent)
        if self.classes == 1:
            prediction = prediction[:, 0]
        return prediction, latent, states, av

    def forward(self, batch):
        prediction, latent, _, _ = self.encode(batch)
        return {"M": prediction, "latent": latent}

    def activity(self):
        return {m: float(a.last_activity) for m, a in self.adapters.items() if a.last_activity is not None}


class Student(nn.Module):
    def __init__(self, dataset, teacher):
        super().__init__()
        self.av = AVOnly(dataset, teacher)
        self.classes = self.av.classes
        self.text_encoder = copy.deepcopy(teacher.core.pretrained_text_encoder).requires_grad_(False)
        self.text_private = copy.deepcopy(teacher.core.encoders["text"]).requires_grad_(True)
        self.ta = copy.deepcopy(teacher.core.pairs["TA"]).requires_grad_(True)
        self.tv = copy.deepcopy(teacher.core.pairs["TV"]).requires_grad_(True)
        self.fusion = copy.deepcopy(teacher.core.trimodal).requires_grad_(True)
        self.head = copy.deepcopy(teacher.core.heads["TAV"]).requires_grad_(True)
        self.text_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    def forward(self, batch, retention):
        av_pred, latent, states, av_state = self.av.encode(batch)
        if retention.shape != (len(av_pred),) or not torch.all((retention >= 0) & (retention <= 1)):
            raise ValueError("Invalid content-retention fraction")
        active = torch.where(retention > 0)[0]
        prediction = av_pred
        if len(active):
            tokens = batch["text"][active]
            mask = tokens[:, 1].bool()
            with torch.no_grad():
                text = self.text_encoder(tokens)
            text = self.text_private(text, mask)
            ta = self.ta(text, states["audio"][active], mask, batch["audio_mask"][active])
            tv = self.tv(text, states["vision"][active], mask, batch["vision_mask"][active])
            tav_pred = self.head(self.fusion(torch.cat([ta, tv, av_state[active]], -1)))
            if self.classes == 1:
                tav_pred = tav_pred[:, 0]
            rho = retention[active] if self.classes == 1 else retention[active, None]
            prediction = av_pred.index_copy(0, active, (1-rho)*av_pred[active] + rho*tav_pred)
        return {"M": prediction, "av": av_pred, "latent": latent, "retention": retention}

    def activity(self):
        return self.av.activity()
