"""Local prepared feature records; no downloads or implicit split reconstruction."""
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .normalization import normalize_records
from .student import DIMS
from .utils import sha

SIZES = {"urfunny": (7614, 980, 994), "mosi": (1284, 229, 686), "sims": (1368, 456, 457)}


class FeatureSplit(Dataset):
    def __init__(self, record, dataset, tokenizer=None):
        self.ids = list(map(str, record["id"]))
        self.y = np.asarray(record["regression_labels"], np.float32).reshape(-1)
        self.values = {m: np.asarray(record[m], np.float32) for m in ("audio", "vision")}
        self.lengths = {m: np.asarray(record[m + "_lengths"], np.int64) for m in self.values}
        if len(self.ids) != len(set(self.ids)) or len(self.ids) != len(self.y) or not np.isfinite(self.y).all():
            raise ValueError("Nonunique IDs or invalid labels")
        if dataset == "urfunny" and not set(np.unique(self.y)) <= {-1., 1.}:
            raise ValueError("UR-FUNNY requires signed labels {-1, +1}, not {0, 1}")
        if dataset == "sims" and np.any(np.abs(self.y) > 1):
            raise ValueError("CH-SIMS labels must be in [-1, 1]")
        for (m, a), width in zip(self.values.items(), DIMS[dataset]):
            n = self.lengths[m]
            if a.ndim != 3 or a.shape[0] != len(self.y) or a.shape[-1] != width or n.shape != (len(self.y),):
                raise ValueError(f"Invalid {m} feature dimensions")
            if not np.isfinite(a).all() or not ((n > 0) & (n <= a.shape[1])).all():
                raise ValueError(f"Invalid {m} values/lengths")
        self.tokens = None
        if tokenizer is not None:
            text = list(map(str, record["raw_text"]))
            if len(text) != len(self.ids):
                raise ValueError("Text and features are not aligned")
            values = tokenizer(text, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
            self.tokens = torch.stack([values["input_ids"], values["attention_mask"]], 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        sample = {"id": self.ids[i], "target": torch.tensor(self.y[i]),
                  **{m: torch.from_numpy(v[i]) for m, v in self.values.items()},
                  **{m + "_mask": torch.arange(v.shape[1]) < self.lengths[m][i] for m, v in self.values.items()}}
        if self.tokens is not None:
            sample["text"] = self.tokens[i]
        return sample


def load_splits(path, dataset, encoder_path=None):
    path = Path(path)
    with path.open("rb") as stream:
        records = pickle.load(stream, encoding="latin1")
    normalization = normalize_records(records) if dataset == "sims" else None
    tokenizer = None
    if encoder_path is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    splits = {s: FeatureSplit(records[s], dataset, tokenizer) for s in ("train", "valid", "test")}
    if tuple(map(len, splits.values())) != SIZES[dataset]:
        raise ValueError(f"Expected split counts {SIZES[dataset]}; do not silently change the protocol")
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if set(splits[a].ids) & set(splits[b].ids):
            raise ValueError("Overlapping split IDs")
    return splits, {"dataset": dataset, "data_sha256": sha(path), "normalization": normalization,
                    "splits": {s: {"ids": ds.ids, "count": len(ds)} for s, ds in splits.items()}}


def loader(split, batch_size=32, train=False, seed=42):
    return DataLoader(split, batch_size=batch_size, shuffle=train, num_workers=0,
                      generator=torch.Generator().manual_seed(seed) if train else None)
