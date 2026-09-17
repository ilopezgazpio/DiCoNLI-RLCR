"""Independent checks of the extracted numerical and buffering responsibilities."""
from types import SimpleNamespace
import pytest
import torch

from rlcr.training.grpo.advantages import grouped_advantages
from rlcr.training.grpo.loss import grpo_loss
from rlcr.training.grpo.rollout_buffer import RolloutBuffer


@pytest.mark.parametrize("scale", [False, True])
def test_advantages_preserve_groups_split_across_workers(scale):
    rewards = torch.arange(8, dtype=torch.float32)[:, None]
    args = SimpleNamespace(num_generations=4, scale_rewards=scale)
    advantages, means, stds = grouped_advantages(
        rewards, torch.ones(1), args, process_index=1, local_count=2
    )
    expected = torch.tensor([0.5, 1.5])
    if scale:
        expected /= torch.arange(4, dtype=torch.float32).std() + 1e-4
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(means, torch.tensor([1.5] * 4 + [5.5] * 4))


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo"])
@pytest.mark.parametrize("beta", [0.0, 0.04])
def test_loss_matches_original_objective(loss_type, beta):
    logps = torch.tensor([[-0.2, -0.5], [-0.9, -0.1]], requires_grad=True)
    old = torch.tensor([[-0.3, -0.1], [-0.2, -0.5]])
    advantages = torch.tensor([0.5, -0.5])
    mask = torch.tensor([[1, 1], [1, 0]])
    reference = torch.tensor([[-0.5, -0.3], [-0.7, -0.2]])
    args = SimpleNamespace(
        epsilon=0.2,
        epsilon_high=0.3,
        delta=1.4,
        beta=beta,
        loss_type=loss_type,
        max_completion_length=4,
    )
    loss, diagnostics = grpo_loss(logps, old, advantages, mask, args, reference)
    ratio = (logps - old).exp()
    terms = -torch.minimum(
        ratio.clamp(max=1.4) * advantages[:, None], ratio.clamp(0.8, 1.3) * advantages[:, None]
    )
    if beta:
        terms += beta * ((reference - logps).exp() - (reference - logps) - 1)
    if loss_type == "grpo":
        expected = ((terms * mask).sum(-1) / mask.sum(-1)).mean()
    elif loss_type == "bnpo":
        expected = (terms * mask).sum() / mask.sum()
    else:
        expected = (terms * mask).sum() / 8
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logps.grad).all()
    assert ("kl" in diagnostics) == bool(beta)


def test_rollouts_are_reused_across_accumulation_and_iterations():
    calls = []

    def generate(examples):
        calls.append(examples)
        return {"tokens": torch.arange(4), "old_logps": None}

    buffer = RolloutBuffer(steps_per_generation=2, num_iterations=2)
    results = [buffer.prepare("batch", generate) for _ in range(5)]
    assert calls == ["batch", "batch"]
    assert results[0] is results[2]
    assert results[1] is results[3]
    assert results[4] is not results[0]
