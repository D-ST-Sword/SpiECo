"""Residual channel-wise PLIF evidence dynamics (SpikingJelly torch backend)."""
import torch
from torch import nn


def temporal_bins(x, mask, steps):
    """Valid-rank binning is invariant to trailing padding and uses no labels."""
    lengths = mask.sum(1).clamp_min(1)
    rank = (mask.long().cumsum(1) - 1).clamp_min(0)
    bins = rank * lengths.clamp_max(steps)[:, None] // lengths[:, None]
    bins = bins.clamp_max(steps - 1)
    assignment = (bins[:, None, :] == torch.arange(steps, device=x.device)[None, :, None]) & mask[:, None, :]
    count = assignment.sum(-1)
    weights = assignment.to(x.dtype) / count.clamp_min(1)[..., None]
    values = torch.bmm(weights, x.masked_fill(~mask[..., None], 0))
    return values, count > 0, bins


class TemporalAdapter(nn.Module):
    def __init__(self, dim=128, mode="lif", steps=8, scale=.1):
        super().__init__()
        if mode != "lif":
            raise ValueError(mode)
        self.mode, self.steps, self.scale = mode, steps, scale
        self.input = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, dim)
        from spikingjelly.activation_based import neuron, surrogate
        self.neuron = neuron.ParametricLIFNode(init_tau=10., decay_input=False,
            v_threshold=1., v_reset=None, surrogate_function=surrogate.ATan(alpha=2.),
            detach_reset=True, step_mode="s", backend="torch")
        # The torch backend broadcasts decay parameters across the batch dimension.
        self.neuron.w = nn.Parameter(self.neuron.w.detach().repeat(dim))
        self.last_activity = None

    def forward(self, x, mask=None):
        static = x.ndim == 2
        if static:
            values = x[:, None, :].expand(-1, self.steps, -1)
            valid = torch.ones(values.shape[:2], dtype=torch.bool, device=x.device)
        else:
            values, valid, bins = temporal_bins(x, mask, self.steps)
        currents = 1.5 * torch.sigmoid(self.norm(self.input(values)))
        self.neuron.reset()
        self.neuron.v = torch.zeros_like(currents[:, 0])
        responses = []
        for t in range(self.steps):
            keep = valid[:, t, None]
            previous = self.neuron.v
            response = self.neuron(currents[:, t])
            self.neuron.v = torch.where(keep, self.neuron.v, previous)
            responses.append(torch.where(keep, response, torch.zeros_like(response)))
        trace = torch.stack(responses, 1)
        self.last_activity = (trace.detach().sum() / (valid.sum().clamp_min(1) * trace.shape[-1]))
        correction = self.output(trace)
        if static:
            return x + self.scale * correction.mean(1)
        expanded = correction.gather(1, bins[..., None].expand(-1, -1, x.shape[-1]))
        return (x + self.scale * expanded).masked_fill(~mask[..., None], 0)
