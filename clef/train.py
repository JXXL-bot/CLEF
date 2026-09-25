from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from clef.model.losses import physical_sdp, spectrum_loss_from_batch


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "spect",
        "true_spect",
        "pred_true_spect",
    }
    return {key: value for key, value in batch.items() if key not in excluded}


def _apply_update(optimizer: torch.optim.Optimizer, model: torch.nn.Module, samples: int) -> None:

    if samples <= 0:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            if parameter.grad.is_sparse:
                parameter.grad._values().div_(samples)
            else:
                parameter.grad.div_(samples)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def run_epoch(
    model: torch.nn.Module,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
    accumulation_steps: int = 1,
) -> dict[str, float | int]:


    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    device = torch.device(device)
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_sdp = 0.0
    total_samples = 0
    pending_samples = 0
    pending_batches = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for raw_batch in batches:
            if not isinstance(raw_batch, Mapping):
                raise TypeError("Each data batch must be a mapping")
            batch = _move_batch(raw_batch, device)
            if "spect" not in batch or not isinstance(batch["spect"], torch.Tensor):
                raise KeyError("Each data batch must contain a tensor named 'spect'")
            target = batch["spect"]
            if target.ndim != 2 or bool((target.sum(dim=1) <= 0).any()):
                raise ValueError("Training and validation require nonempty observed spectra")
            output = model(**_model_inputs(batch))
            if not isinstance(output, Mapping) or "spect" not in output:
                raise TypeError("Model must return a mapping containing 'spect'")

            loss = spectrum_loss_from_batch(output, batch)
            batch_size = int(target.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            with torch.no_grad():
                total_sdp += float(physical_sdp(output["spect"].detach(), target).sum().item())
            total_samples += batch_size

            if training:
                (loss * batch_size).backward()
                pending_samples += batch_size
                pending_batches += 1
                if pending_batches == accumulation_steps:
                    _apply_update(optimizer, model, pending_samples)
                    pending_samples = 0
                    pending_batches = 0

    if training and pending_batches:
        _apply_update(optimizer, model, pending_samples)
    if total_samples == 0:
        raise ValueError("Data loader yielded no spectra")
    return {
        "loss": total_loss / total_samples,
        "sdp": total_sdp / total_samples,
        "samples": total_samples,
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    validation_sdp: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_kwargs": {
                    key: str(value) if isinstance(value, float) else value
                    for key, value in getattr(model, "model_kwargs", {}).items()
                },
                "validation_sdp": str(validation_sdp),
            },
            temporary,
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def fit(
    model: torch.nn.Module,
    train_loader: Iterable[Mapping[str, Any]],
    validation_loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | str,
    *,
    accumulation_steps: int = 1,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:


    if epochs < 1:
        raise ValueError("epochs must be positive")
    device = torch.device(device)
    model.to(device)
    history: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_sdp = float("-inf")
    for epoch in range(1, epochs + 1):
        train_result = run_epoch(
            model, train_loader, device, optimizer, accumulation_steps
        )
        validation_result = run_epoch(model, validation_loader, device)
        history.append(
            {"epoch": epoch, "train": train_result, "validation": validation_result}
        )
        validation_sdp = float(validation_result["sdp"])
        if validation_sdp > best_sdp:
            best_sdp = validation_sdp
            best_epoch = epoch
            if checkpoint_path is not None:
                _save_checkpoint(
                    Path(checkpoint_path), model, epoch, validation_sdp
                )
    return {"best_epoch": best_epoch, "best_validation_sdp": best_sdp, "history": history}
