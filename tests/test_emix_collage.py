"""Behavioral checks for object paste and exact dual-branch loss semantics."""
import torch

from embodied_mixup.policies.DP.emix_diffusion import combine_emix_losses
from embodied_mixup.policies.utils.emix_loader_registry_utils import paste_selected_object


def test_collage_pastes_only_selected_pixels_and_all_channels():
    anchor = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    donor = anchor + 100
    mask = torch.tensor([[False, True], [False, False]])
    result = paste_selected_object(anchor, donor, mask)
    assert torch.equal(result[:, 0, 1], donor[:, 0, 1])
    assert torch.equal(result[:, 1, 0], anchor[:, 1, 0])
    assert torch.equal(result[:, 1, 1], anchor[:, 1, 1])


def test_collage_loss_is_sum_not_weighted_average():
    loss_0 = torch.tensor([[[2.0], [100.0]]])
    loss_1 = torch.tensor([[[3.0], [200.0]]])
    valid = torch.tensor([[True, False]])
    alpha = torch.tensor([0.25])
    assert combine_emix_losses(loss_0, loss_1, valid, alpha, "collage").item() == 5.0
    assert combine_emix_losses(loss_0, loss_1, valid, alpha, "overlay").item() == 2.75
