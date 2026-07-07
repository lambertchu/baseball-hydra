"""Tests for the Phase 3 GRU ROS model."""

from __future__ import annotations

import torch

from src.models.mtl_ros.model import MTLQuantileNetwork
from src.models.ros.model import ROSSequenceNetwork


def _model() -> ROSSequenceNetwork:
    base = MTLQuantileNetwork(
        n_features=5,
        n_targets=6,
        n_quantiles=5,
        hidden_dims=[12, 8],
        head_dim=4,
        dropouts=[0.0, 0.0],
        two_stage=True,
        speed_head_indices=[5],
        taus=[0.05, 0.25, 0.5, 0.75, 0.95],
    )
    return ROSSequenceNetwork(
        phase2_network=base,
        seq_group_dims={"mechanics": 3, "plate": 4, "outcome": 5},
        encoder_dim=6,
        gru_hidden_dim=10,
        dropout=0.0,
    )


def test_sequence_network_outputs_all_timesteps() -> None:
    model = _model()
    model.eval()

    outputs = model(
        seq=torch.randn(4, 6, 12),
        phase2_x=torch.randn(4, 5),
        seq_mask=torch.ones(4, 6, dtype=torch.bool),
        blend_features=torch.rand(4, 6, 2),
    )

    assert outputs["quantiles"].shape == (4, 6, 6, 5)
    assert outputs["pa_remaining"].shape == (4, 6, 1)
    assert outputs["latent"].shape == (4, 6, 8)


def test_phase2_network_is_frozen_during_backprop() -> None:
    model = _model()
    model.train()

    outputs = model(
        seq=torch.randn(3, 4, 12),
        phase2_x=torch.randn(3, 5),
        seq_mask=torch.ones(3, 4, dtype=torch.bool),
        blend_features=torch.rand(3, 4, 2),
    )
    loss = outputs["quantiles"][:, -1].sum() + outputs["pa_remaining"][:, -1].sum()
    loss.backward()

    assert all(not p.requires_grad for p in model.phase2_network.parameters())
    assert all(p.grad is None for p in model.phase2_network.parameters())
    assert any(p.grad is not None for p in model.gru.parameters())
