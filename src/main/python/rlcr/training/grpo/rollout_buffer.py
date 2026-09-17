"""Reuse a generation batch across accumulation steps and GRPO iterations."""

from .tensors import shuffle_tensor_dict, split_tensor_dict


class RolloutBuffer:
    def __init__(self, steps_per_generation, num_iterations):
        self.steps_per_generation = steps_per_generation
        self.num_iterations = num_iterations
        self.step = 0
        self.batches = None

    def prepare(self, examples, generate):
        interval = self.steps_per_generation * self.num_iterations
        if self.step % interval == 0 or self.batches is None:
            rollout = shuffle_tensor_dict(generate(examples))
            self.batches = split_tensor_dict(rollout, self.steps_per_generation)
        batch = self.batches[self.step % self.steps_per_generation]
        self.step += 1
        return batch
