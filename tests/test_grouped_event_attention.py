from __future__ import annotations

import math

import torch

from clef.model.spectrum_model import CompositionEventAttention


def _reference(module, formula, events, indices, mask):
    batch_size, formula_count, _ = formula.shape
    event_count = events.shape[1]
    queries = module.query(module.formula_norm(formula)).reshape(
        batch_size, formula_count, module.heads, module.head_width)
    keys = module.key(module.event_norm(events)).reshape(
        batch_size, event_count, module.heads, module.head_width)
    values = module.value(module.event_norm(events)).reshape(
        batch_size, event_count, module.heads, module.head_width)
    aggregate = torch.zeros_like(queries)
    for batch in range(batch_size):
        for formula_index in range(formula_count):
            members = mask[batch].bool() & (indices[batch] == formula_index)
            if not bool(members.any()):
                continue
            logits = (
                queries[batch, formula_index].unsqueeze(0) * keys[batch, members]
            ).sum(dim=-1) / math.sqrt(module.head_width)
            attention = torch.softmax(logits, dim=0)
            aggregate[batch, formula_index] = (
                attention.unsqueeze(-1) * values[batch, members]
            ).sum(dim=0)
    return formula + module.output(
        aggregate.reshape(batch_size, formula_count, -1))


def test_grouped_attention_matches_reference_and_gradients():
    torch.manual_seed(17)
    module = CompositionEventAttention(
        formula_width=8, event_width=16, heads=2, head_width=4, dropout=0.0)
    with torch.no_grad():
        torch.nn.init.normal_(module.output.weight, std=0.1)
        module.output.bias.fill_(0.25)
    formula = torch.randn(2, 4, 8, requires_grad=True)
    events = torch.randn(2, 7, 16, requires_grad=True)
    indices = torch.tensor([
        [0, 2, 2, -1, 4, 0, 1],
        [-1, 0, 0, 1, 2, 3, 3],
    ])
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.bool)

    actual = module(formula, events, indices, mask)
    expected = _reference(module, formula, events, indices, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    sources = (formula, events, module.query.weight, module.key.weight,
               module.value.weight, module.output.weight)
    actual_gradients = torch.autograd.grad(actual.square().sum(), sources,
                                           retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), sources)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-4, atol=1e-6)


def test_grouped_attention_handles_no_matching_events():
    module = CompositionEventAttention(
        formula_width=8, event_width=16, heads=2, head_width=4, dropout=0.0)
    with torch.no_grad():
        module.output.bias.fill_(0.5)
    formula = torch.randn(1, 3, 8, requires_grad=True)
    events = torch.randn(1, 2, 16, requires_grad=True)
    indices = torch.tensor([[-1, 99]])
    mask = torch.ones((1, 2), dtype=torch.bool)
    output = module(formula, events, indices, mask)
    torch.testing.assert_close(output, formula + 0.5)
    output.square().sum().backward()
    assert torch.isfinite(formula.grad).all()
    assert events.grad is not None and torch.count_nonzero(events.grad) == 0
