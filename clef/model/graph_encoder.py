from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def create_nonlin(name: str) -> nn.Module:
    name = str(name).lower()
    if name in {"leakyrelu", "leaky_relu"}:
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


def parse_agg_func(value):

    if value is None:
        return "sum"
    if callable(value):
        return value
    value = str(value).lower()
    if value in {"sum", "mean", "max", "goodmax"}:
        return value
    raise ValueError(f"Unknown graph aggregation: {value}")


class MaskedBatchNorm1d(nn.Module):


    def __init__(self, features: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(int(features))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        original_shape = values.shape
        flat = values.reshape(-1, original_shape[-1])
        valid = mask.reshape(-1).bool()
        output = torch.zeros_like(flat)
        if valid.any():
            selected = flat[valid]
            if self.training and selected.shape[0] < 2:

                selected = F.batch_norm(
                    selected, self.norm.running_mean, self.norm.running_var,
                    self.norm.weight, self.norm.bias, training=False,
                    eps=self.norm.eps,
                )
            else:
                selected = self.norm(selected)
            output[valid] = selected
        return output.reshape(original_shape)


class MaskedLayerNorm1d(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(features))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.norm(values) * mask.reshape(values.shape[:-1]).unsqueeze(-1).to(values.dtype)


class GraphMatLayerFast3(nn.Module):


    def __init__(self, input_feature_n: int, output_feature_n: int,
                 GS: int = 4, noise: float = 0.0, agg_func="sum",
                 use_bias: bool = True, nonlin: str = "leakyrelu",
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.relation_count = int(GS)
        self.aggregation = parse_agg_func(agg_func)
        self.self_linear = nn.Linear(input_feature_n, output_feature_n, bias=use_bias)
        self.relation_linear = nn.ModuleList(
            nn.Linear(input_feature_n, output_feature_n, bias=False)
            for _ in range(self.relation_count)
        )
        self.activation = create_nonlin(nonlin)
        self.dropout = nn.Dropout(float(dropout))
        for linear in [self.self_linear, *self.relation_linear]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, adjacency: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim != 4 or values.ndim != 3:
            raise ValueError("Expected adjacency [B,R,N,N] and atom states [B,N,F]")
        if adjacency.shape[1] != self.relation_count:
            raise ValueError("Bond-relation channel count does not match the encoder")
        if adjacency.shape[0] != values.shape[0] or adjacency.shape[2:] != values.shape[1:2] * 2:
            raise ValueError("Adjacency and atom state shapes do not match")
        edges = adjacency.to(values.dtype)
        degree = edges.sum(dim=(1, 3)).clamp_min(1.0)
        scale = degree.rsqrt()
        normalized = edges * scale[:, None, :, None] * scale[:, None, None, :]
        messages = []
        for relation, linear in enumerate(self.relation_linear):
            messages.append(torch.bmm(normalized[:, relation], linear(values)))
        if callable(self.aggregation):
            combined = self.aggregation(torch.stack(messages, dim=0))
        elif self.aggregation in {"max", "goodmax"}:
            combined = torch.stack(messages, dim=0).max(dim=0).values
        elif self.aggregation == "mean":
            combined = torch.stack(messages, dim=0).mean(dim=0)
        else:
            combined = torch.stack(messages, dim=0).sum(dim=0)
        return self.dropout(self.activation(self.self_linear(values) + combined))


class GraphMatLayersNormAfterRes(nn.Module):


    def __init__(self, input_feature_n: int, output_features_n: Sequence[int],
                 resnet: bool = True, GS: int = 4, norm: Optional[str] = "layer",
                 force_use_bias: bool = False, noise: float = 0.0,
                 agg_func="sum", layer_class: str = "GraphMatLayerFast3",
                 layer_config: Optional[dict] = None, **kwargs):
        super().__init__()
        layer_type = {"GraphMatLayerFast3": GraphMatLayerFast3}.get(layer_class)
        if layer_type is None:
            raise ValueError(f"Unknown graph layer: {layer_class}")
        self.resnet = bool(resnet)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        previous = int(input_feature_n)
        for width in output_features_n:
            width = int(width)
            self.layers.append(layer_type(
                previous, width, GS=GS, noise=noise, agg_func=agg_func,
                use_bias=(norm is None or force_use_bias),
                **dict(layer_config or {}),
            ))
            if norm == "layer":
                self.norms.append(MaskedLayerNorm1d(width))
            elif norm == "batch":
                self.norms.append(MaskedBatchNorm1d(width))
            elif norm is None:
                self.norms.append(nn.Identity())
            else:
                raise ValueError(f"Unknown graph normalization: {norm}")
            previous = width

    def forward(self, adjacency: torch.Tensor, values: torch.Tensor,
                input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_mask is None:
            input_mask = torch.ones(values.shape[:2], dtype=values.dtype, device=values.device)
        for layer_index, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            updated = layer(adjacency, values)
            if self.resnet and layer_index > 0 and updated.shape == values.shape:
                updated = updated + values
            if not isinstance(norm, nn.Identity):
                updated = norm(updated, input_mask)
            values = updated * input_mask.unsqueeze(-1).to(updated.dtype)
        return values
