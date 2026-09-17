"""Opt-in two-worker CPU DDP regression, runnable through torchrun + pytest."""
import os

from datasets import Dataset
from peft import LoraConfig
import pytest
import torch
import torch.distributed as dist

from rlcr.arguments.grpo_config import GRPOConfig
from rlcr.training.grpo.trainer import GRPOTrainer


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="Requires two torchrun workers"
)
@pytest.mark.parametrize("peft", [False, True], ids=["full", "lora"])
def test_two_workers_share_grpo_groups_and_update_identically(tiny_model, tmp_path, peft):
    model, tokenizer = tiny_model
    args = GRPOConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        beta=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_steps=2,
        max_prompt_length=32,
        max_completion_length=4,
        learning_rate=0.01,
        report_to=[],
        save_strategy="no",
        log_completions=False,
        scale_rewards=False,
        disable_tqdm=True,
        ddp_find_unused_parameters=False,
    )

    def rewards(completions, **kwargs):
        # Each worker owns only two of this question's four completions.
        return [float(dist.get_rank() * 2 + index) for index in range(len(completions))]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        reward_funcs=[rewards],
        train_dataset=Dataset.from_dict(
            {"prompt": [[{"role": "user", "content": "question"}]] * 2}
        ),
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj", "v_proj"])
        if peft
        else None,
    )
    before = {
        name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
    }
    result = trainer.train()
    assert result.global_step == 2
    assert trainer.accelerator.num_processes == 2
    assert isinstance(trainer.model_wrapped, torch.nn.parallel.DistributedDataParallel)
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    )
    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in trainer.model.named_parameters()
        if not parameter.requires_grad
    )
    local_advantages = (
        torch.cat([batch["advantages"] for batch in trainer.rollout_buffer.batches]).sort().values
    )
    expected = torch.tensor([-1.5, -0.5]) if dist.get_rank() == 0 else torch.tensor([0.5, 1.5])
    torch.testing.assert_close(local_advantages, expected)
    for parameter in trainer.model.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
