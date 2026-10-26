"""Sample-ID-stable content erasure used only during consolidation training."""
import hashlib
import numpy as np
import torch

GRID = tuple(i / 10 for i in range(11))
EVAL_SEED = 20261002


def erase(tokens, ids, rates, seed, special_ids=(0, 1, 2), pad_id=1):
    if tokens.device.type != "cpu":
        raise ValueError("Apply erasure to raw CPU token IDs before encoding")
    if tokens.ndim != 3 or tokens.shape[1] != 2 or len(ids) != len(tokens):
        raise ValueError("Expected [batch, IDs/mask, sequence] and one ID per sample")
    rates = np.broadcast_to(np.asarray(rates, dtype=float), (len(tokens),))
    if not np.isfinite(rates).all() or (rates < 0).any() or (rates > 1).any():
        raise ValueError("Erasure rates must be in [0, 1]")
    output = tokens.clone()
    kept, counts, erased = [], [], []
    for i, (uid, rate) in enumerate(zip(ids, rates)):
        content = tokens[i, 1].bool().clone()
        for special in special_ids:
            content &= tokens[i, 0] != special
        positions = torch.where(content)[0].numpy()
        count = len(positions)
        key = hashlib.sha256(f"text-erasure-v1|{seed}|{uid}".encode()).digest()
        order = np.random.default_rng(int.from_bytes(key[:8], "little")).permutation(positions)
        removed = min(count, int(np.floor(float(rate)*count + .5)))
        if removed:
            index = torch.from_numpy(order[:removed].copy())
            output[i, 0, index] = pad_id
            output[i, 1, index] = 0
        kept.append((count-removed)/count if count else 0.)
        counts.append(count)
        erased.append(removed)
    return output, torch.tensor(kept, dtype=torch.float32), np.asarray(counts), np.asarray(erased)
