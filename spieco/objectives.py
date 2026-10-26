"""Consolidation objective and the locked AV refinement operations."""
import torch
from torch import nn
from torch.nn import functional as F


def consolidation_loss(output, target, teacher_prediction, teacher_latent):
    terms = {"task": F.l1_loss(output["M"], target),
             "av_task": F.l1_loss(output["av"], target),
             "prediction_kd": F.smooth_l1_loss(output["av"], teacher_prediction),
             "latent_kd": (1 - F.cosine_similarity(output["latent"], teacher_latent, dim=-1)).mean()}
    total = terms["task"] + .5 * terms["av_task"] + .5 * terms["prediction_kd"] + .1 * terms["latent_kd"]
    return total, terms


def set_refinement_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = .5
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = .5


def zero_augmentation(batch):
    # Retain RNG calls from the original zero-noise/channel-drop/frame-drop recipe.
    result = dict(batch)
    for m in ("audio", "vision"):
        value = batch[m]
        channel = torch.rand((len(value), 1, value.shape[2]), device=value.device) >= 0.
        frame = torch.rand((*value.shape[:2], 1), device=value.device) >= 0.
        noise = torch.randn_like(value) * 0.
        candidate = (value + noise) * channel * frame
        result[m] = torch.where(batch[m + "_mask"][..., None], candidate, value)
    return result
