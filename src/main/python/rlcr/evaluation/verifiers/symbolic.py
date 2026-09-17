"""Evaluate answers with math verification and normalized exact matching."""
from .correctness import gen_correctness_reward
from .reporting import build_report


def confidence_verifier(local_dataset, config, **kwargs):
    evals = []
    for example in local_dataset:
        completions = [
            [{"role": "assistant", "content": example[f"{config.name}-output_{index}"]}]
            for index in range(config.n)
        ]
        scores = gen_correctness_reward(completions, [example["answer"]] * config.n)
        evals.append([int(score == 1) for score in scores])
    return build_report(local_dataset, config, evals)
