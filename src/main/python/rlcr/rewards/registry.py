"""Select the experiment's reward functions without coupling them to training."""
from functools import partial
from .accuracy import accuracy_reward
from .brier import brier_reward
from .confidence import confidence_one_or_zero, mean_confidence_reward
from .format import format_reward


def build_reward_functions(script_args):
    registry = {
        "format": partial(format_reward, format_pattern=script_args.format_pattern),
        "accuracy": partial(accuracy_reward, format_pattern=script_args.format_pattern),
        "brier": partial(brier_reward, format_pattern=script_args.format_pattern),
        "mean_confidence": mean_confidence_reward,
        "confidence_one_or_zero": confidence_one_or_zero,
    }
    unknown = set(script_args.reward_funcs) - registry.keys()
    if unknown:
        raise ValueError(
            f"Unknown reward functions: {', '.join(sorted(unknown))}. "
            f"Supported: {', '.join(sorted(registry))}."
        )
    return [registry[name] for name in script_args.reward_funcs]
