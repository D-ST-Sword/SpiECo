import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(2)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def device_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_state(model, path, dataset=None, data_sha256=None):
    # Load only checkpoints you trust: historical torch checkpoint files use pickle.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if dataset is not None and saved.get("dataset") != dataset:
        raise ValueError("Checkpoint dataset mismatch")
    if data_sha256 is not None and saved.get("data_sha256") != data_sha256:
        raise ValueError("Checkpoint feature/split file mismatch")
    model.load_state_dict(saved["model"], strict=True)
    return saved
