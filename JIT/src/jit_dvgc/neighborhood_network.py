"""Torch neighborhood encoder; raw causal observations remain in PPO storage."""
import torch
from torch import nn


class NeighborhoodNetwork(nn.Module):
    """Shared row MLP and masked mean pooling followed by an existing RSL head.

    Actor and critic each own an instance. Invalid rows are zeroed before the
    MLP as well as after it, so even arbitrary/nonfinite padding is immaterial.
    """

    def __init__(self, config, head):
        super().__init__()
        self.config = dict(config)
        width = config['summary_dim']
        self.encoder = nn.Sequential(
            nn.Linear(config['feature_dim'] - 1, width), nn.ELU(),
            nn.Linear(width, width), nn.ELU())
        self.head = head

    def forward(self, obs):
        c = self.config
        end = c['base_dim'] + c['neighbors'] * c['feature_dim']
        rows = obs[..., c['base_dim']:end].reshape(
            *obs.shape[:-1], c['neighbors'], c['feature_dim'])
        valid = rows[..., -1:] > 0
        features = torch.where(valid, rows[..., :-1], 0.)
        encoded = self.encoder(features)
        summary = torch.where(valid, encoded, 0.).sum(dim=-2)
        summary = summary / valid.sum(dim=-2).clamp(min=1)
        return self.head(torch.cat((obs[..., :c['base_dim']], summary, obs[..., end:]), dim=-1))
