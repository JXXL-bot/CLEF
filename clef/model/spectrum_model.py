from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .graph_encoder import GraphMatLayersNormAfterRes, MaskedBatchNorm1d


FORMULA_BLOCK_WIDTHS = (50, 46, 30, 30, 30, 30, 30, 30)


def encode_formula_counts(counts: torch.Tensor) -> torch.Tensor:

    if counts.shape[-1] != len(FORMULA_BLOCK_WIDTHS):
        raise ValueError("Formula counts must have eight element columns")
    blocks = []
    for element, width in enumerate(FORMULA_BLOCK_WIDTHS):
        levels = torch.arange(width, device=counts.device)
        blocks.append((levels <= counts[..., element, None]).to(torch.float32))
    return torch.cat(blocks, dim=-1)


class EventSetContext(nn.Module):


    def __init__(self, width: int = 256, latent_count: int = 32,
                 attention_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_count, width) * 0.02)
        self.latent_norm = nn.LayerNorm(width)
        self.event_norm = nn.LayerNorm(width)
        self.latent_reads = nn.MultiheadAttention(
            width, attention_heads, dropout=dropout, batch_first=True)
        self.event_reads = nn.MultiheadAttention(
            width, attention_heads, dropout=dropout, batch_first=True)
        self.latent_ff_norm = nn.LayerNorm(width)
        self.event_ff_norm = nn.LayerNorm(width)
        self.latent_ff = nn.Sequential(
            nn.Linear(width, 4 * width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4 * width, width), nn.Dropout(dropout))
        self.event_ff = nn.Sequential(
            nn.Linear(width, 4 * width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4 * width, width), nn.Dropout(dropout))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))

    def forward(self, events: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        if events.shape[1] == 0:
            return events
        original = events
        valid = event_mask.bool()
        all_empty = ~valid.any(dim=1)
        if all_empty.any():
            valid = valid.clone()
            valid[all_empty, 0] = True
        latent = self.latents.unsqueeze(0).expand(events.shape[0], -1, -1)
        latent_read, _ = self.latent_reads(
            self.latent_norm(latent), self.event_norm(events),
            self.event_norm(events), key_padding_mask=~valid,
            need_weights=False)
        latent = latent + latent_read
        latent = latent + self.latent_ff(self.latent_ff_norm(latent))
        event_read, _ = self.event_reads(
            self.event_norm(events), self.latent_norm(latent),
            self.latent_norm(latent), need_weights=False)
        events = events + event_read
        events = events + self.event_ff(self.event_ff_norm(events))
        return (original + self.output(events)) * event_mask.unsqueeze(-1).to(events.dtype)


class CompositionEventAttention(nn.Module):


    def __init__(self, formula_width: int = 8, event_width: int = 256,
                 heads: int = 8, head_width: int = 16, dropout: float = 0.1):
        super().__init__()
        self.heads = int(heads)
        self.head_width = int(head_width)
        total = self.heads * self.head_width
        self.formula_norm = nn.LayerNorm(formula_width)
        self.event_norm = nn.LayerNorm(event_width)
        self.query = nn.Linear(formula_width, total)
        self.key = nn.Linear(event_width, total)
        self.value = nn.Linear(event_width, total)
        self.output = nn.Linear(total, formula_width)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, formula: torch.Tensor, events: torch.Tensor,
                event_formula_index: torch.Tensor,
                event_mask: torch.Tensor) -> torch.Tensor:
        batch_size, formula_count, _ = formula.shape
        event_count = events.shape[1]
        queries = self.query(self.formula_norm(formula)).reshape(
            batch_size, formula_count, self.heads, self.head_width)
        keys = self.key(self.event_norm(events)).reshape(
            batch_size, event_count, self.heads, self.head_width)
        values = self.value(self.event_norm(events)).reshape(
            batch_size, event_count, self.heads, self.head_width)
        valid = event_mask.bool() & (event_formula_index >= 0)
        valid = valid & (event_formula_index < formula_count)
        safe_index = event_formula_index.long().clamp(0, formula_count - 1)
        event_queries = queries.gather(
            1, safe_index[:, :, None, None].expand(
                -1, -1, self.heads, self.head_width))
        logits = (event_queries * keys).sum(dim=-1) / math.sqrt(self.head_width)


        group_ids = (
            torch.arange(batch_size, device=formula.device)[:, None, None]
            * (formula_count * self.heads)
            + safe_index[:, :, None] * self.heads
            + torch.arange(self.heads, device=formula.device)[None, None, :]
        ).expand(-1, event_count, -1)
        valid_heads = valid[:, :, None].expand(-1, -1, self.heads)
        selected_groups = group_ids[valid_heads]
        selected_logits = logits[valid_heads]
        selected_values = values[valid_heads]
        group_count = batch_size * formula_count * self.heads
        maxima = selected_logits.new_full((group_count,), -torch.inf)
        maxima.scatter_reduce_(
            0, selected_groups, selected_logits, reduce="amax", include_self=True)
        exponentials = torch.exp(selected_logits - maxima[selected_groups])
        denominators = selected_logits.new_zeros((group_count,))
        denominators.scatter_add_(0, selected_groups, exponentials)
        attention = self.dropout(
            exponentials / denominators[selected_groups])
        aggregate = values.new_zeros((group_count, self.head_width))
        aggregate.index_add_(
            0, selected_groups, attention[:, None] * selected_values)
        aggregate = aggregate.reshape(
            batch_size, formula_count, self.heads, self.head_width)
        update = self.output(aggregate.reshape(batch_size, formula_count, -1))
        return formula + update


class CLEFSpectrumModel(nn.Module):


    def __init__(self, atom_feature_dim: int = 45, relation_count: int = 4,
                 graph_width: int = 512, graph_layers: int = 16,
                 event_feature_dim: int = 45, event_width: int = 256,
                 latent_count: int = 32, dropout: float = 0.1,
                 spectrum_bins: int = 512):
        super().__init__()
        dropout = float(dropout)
        self.model_kwargs = {
            "atom_feature_dim": int(atom_feature_dim),
            "relation_count": int(relation_count),
            "graph_width": int(graph_width),
            "graph_layers": int(graph_layers),
            "event_feature_dim": int(event_feature_dim),
            "event_width": int(event_width),
            "latent_count": int(latent_count),
            "dropout": float(dropout),
            "spectrum_bins": int(spectrum_bins),
        }
        self.spectrum_bins = int(spectrum_bins)
        self.input_norm = MaskedBatchNorm1d(atom_feature_dim)
        self.graph = GraphMatLayersNormAfterRes(
            atom_feature_dim, [graph_width] * graph_layers,
            resnet=True, GS=relation_count, norm="layer",
            layer_config={"dropout": dropout})
        event_input = event_feature_dim + 3 * graph_width
        self.event_encoder = nn.Sequential(
            nn.Linear(event_input, event_width), nn.LayerNorm(event_width),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(event_width, event_width), nn.LayerNorm(event_width),
            nn.ReLU(), nn.Dropout(dropout))
        self.event_context = EventSetContext(
            event_width, latent_count, attention_heads=8, dropout=dropout)
        self.formula_encoder = nn.Linear(sum(FORMULA_BLOCK_WIDTHS), 8)
        self.formula_event_attention = CompositionEventAttention(
            formula_width=8, event_width=event_width, heads=8,
            head_width=16, dropout=dropout)
        self.atom_key = nn.Linear(graph_width, 8)
        self.formula_gru = nn.ModuleList(
            nn.GRUCell(sum(FORMULA_BLOCK_WIDTHS), graph_width)
            for _ in range(3))
        score_layers = [nn.Linear(graph_width, 128), nn.ReLU()]
        for _ in range(9):
            score_layers.extend([nn.Linear(128, 128), nn.ReLU()])
        score_layers.extend([nn.LayerNorm(128), nn.Linear(128, 1)])
        self.scorer = nn.Sequential(*score_layers)

    @staticmethod
    def _boundary_states(atom_states: torch.Tensor,
                         atom_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, event_count, slots = atom_indices.shape
        atom_count = atom_states.shape[1]
        valid = (atom_indices >= 0) & (atom_indices < atom_count)
        safe = atom_indices.clamp(0, max(atom_count - 1, 0))
        batch_index = torch.arange(batch_size, device=atom_states.device)
        batch_index = batch_index[:, None, None].expand(-1, event_count, slots)
        picked = atom_states[batch_index, safe]
        picked = picked * valid.unsqueeze(-1).to(picked.dtype)
        denominator = valid.sum(dim=2, keepdim=True).clamp_min(1)
        mean = picked.sum(dim=2) / denominator
        maximum = picked.masked_fill(~valid.unsqueeze(-1), torch.finfo(picked.dtype).min)
        maximum = maximum.max(dim=2).values
        maximum = torch.where(valid.any(dim=2, keepdim=True), maximum, torch.zeros_like(maximum))
        return mean, maximum

    @staticmethod
    def _match_events(formula_counts: torch.Tensor, formula_mask: torch.Tensor,
                      event_counts: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:

        widths = FORMULA_BLOCK_WIDTHS
        strides = []
        product = 1
        for width in widths:
            strides.append(product)
            product *= width
        scale = torch.tensor(strides, dtype=torch.int64, device=formula_counts.device)
        formula_codes = (formula_counts.long() * scale).sum(dim=-1)
        event_codes = (event_counts.long() * scale).sum(dim=-1)
        sentinel = torch.iinfo(torch.int64).max
        formula_codes = torch.where(formula_mask.bool(), formula_codes, sentinel)
        sorted_codes, order = formula_codes.sort(dim=1)
        positions = torch.searchsorted(sorted_codes.contiguous(), event_codes.contiguous())
        positions = positions.clamp(max=max(formula_counts.shape[1] - 1, 0))
        found = sorted_codes.gather(1, positions) == event_codes
        found = found & event_mask.bool()
        matched = order.gather(1, positions)
        return torch.where(found, matched, -torch.ones_like(matched))

    def forward(self, adj: torch.Tensor, vect_feat: torch.Tensor,
                input_mask: torch.Tensor, formula_counts: torch.Tensor,
                formula_mask: torch.Tensor, isotope_mass_idx: torch.Tensor,
                isotope_intensity: torch.Tensor, event_features: torch.Tensor,
                event_atom_idx: torch.Tensor, event_mask: torch.Tensor,
                event_formula_index: Optional[torch.Tensor] = None,
                event_counts: Optional[torch.Tensor] = None,
                **kwargs) -> dict[str, torch.Tensor]:
        if formula_counts.shape[1] == 0:
            raise ValueError("No candidate formulas were provided")
        atom_mask = input_mask.to(vect_feat.dtype)
        normalized = self.input_norm(vect_feat, atom_mask)
        atom_states = self.graph(adj, normalized, atom_mask)
        atom_states = atom_states * atom_mask.unsqueeze(-1)
        molecule_state = atom_states.sum(dim=1) / atom_mask.sum(dim=1, keepdim=True).clamp_min(1)
        boundary_mean, boundary_max = self._boundary_states(atom_states, event_atom_idx.long())
        event_input = torch.cat([
            event_features.float(),
            molecule_state.unsqueeze(1).expand(-1, event_features.shape[1], -1),
            boundary_mean, boundary_max], dim=-1)
        encoded_events = self.event_encoder(event_input)
        encoded_events = self.event_context(encoded_events, event_mask)
        formula_encoding = encode_formula_counts(formula_counts.long())
        formula_tokens = self.formula_encoder(formula_encoding)
        if event_formula_index is None:
            if event_counts is None:


                precursor_atoms = atom_mask.sum(dim=1)[:, None, None]
                event_counts = torch.round(
                    event_features[..., 11:19] * precursor_atoms).long()
            event_formula_index = self._match_events(
                formula_counts, formula_mask, event_counts, event_mask)
        formula_tokens = self.formula_event_attention(
            formula_tokens, encoded_events, event_formula_index, event_mask)


        atom_logits = torch.einsum("bfa,bna->bfn", formula_tokens, self.atom_key(atom_states))
        atom_weights = torch.softmax(atom_logits, dim=-1)
        formula_state = torch.einsum("bfn,bnd->bfd", atom_weights, atom_states)
        for cell in self.formula_gru:
            formula_state = cell(
                formula_encoding.reshape(-1, formula_encoding.shape[-1]),
                formula_state.reshape(-1, formula_state.shape[-1]),
            ).reshape_as(formula_state)
        scores = self.scorer(formula_state).squeeze(-1)
        scores = scores.masked_fill(~formula_mask.bool(), -100.0)
        formula_weight = torch.softmax(scores, dim=1)
        valid_peak = (isotope_mass_idx >= 0) & (isotope_mass_idx < self.spectrum_bins)
        peak_values = (formula_weight.unsqueeze(-1) * isotope_intensity.float())
        peak_values = peak_values * valid_peak.to(peak_values.dtype)
        spectrum = peak_values.new_zeros((peak_values.shape[0], self.spectrum_bins))
        spectrum.scatter_add_(
            1, isotope_mass_idx.long().clamp(0, self.spectrum_bins - 1).flatten(1),
            peak_values.flatten(1))
        return {"spect": spectrum, "formula_weight": formula_weight,
                "event_formula_index": event_formula_index}
