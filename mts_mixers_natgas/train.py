"""Training loop shared by every backbone/strategy/channel-set combination.

Takes an already-constructed `Forecaster` rather than building one internally: the caller picks
the backbone, strategy, channel count, and mixer hyperparameters, which vary across the
baseline comparison and the channel-ablation ladder in cli.py.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mts_mixers_natgas.dataset import WindowDataset
from mts_mixers_natgas.models import Forecaster

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 6
    seed: int = 0


@dataclass(frozen=True)
class TrainResult:
    model: Forecaster
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = -1


def train_forecaster(
    model: Forecaster,
    train_dataset: WindowDataset,
    val_dataset: WindowDataset,
    config: TrainConfig | None = None,
    *,
    run_name: str = "run",
) -> TrainResult:
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(config.epochs):
        train_losses.append(_run_epoch(model, train_loader, optimizer))
        val_losses.append(_run_epoch(model, val_loader, optimizer=None))
        LOGGER.info(
            "%s epoch=%d train_loss=%.5f val_loss=%.5f",
            run_name,
            epoch,
            train_losses[-1],
            val_losses[-1],
        )

        if val_losses[-1] < best_val_loss - 1e-6:
            best_val_loss = val_losses[-1]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                LOGGER.info(
                    "%s: early stopping at epoch %d (best epoch %d)",
                    run_name,
                    epoch,
                    best_epoch,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
    )


def _run_epoch(
    model: Forecaster,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_count = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for x, y in loader:
            if optimizer is not None:
                optimizer.zero_grad()
            prediction = model(x)
            loss = F.mse_loss(prediction, y)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)
    return total_loss / total_count
