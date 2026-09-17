"""Select dataset-neutral reward functions."""
from .accuracy import accuracy_reward
from .brier import brier_reward
from .format import format_reward


def build_reward_functions(script_args):
    registry = {
        "format": format_reward,
        "accuracy": accuracy_reward,
        "brier": brier_reward,
    }
    unknown = set(script_args.reward_funcs) - registry.keys()
    if unknown:
        raise ValueError(
            f"Unknown reward functions: {', '.join(sorted(unknown))}. "
            f"Supported: {', '.join(sorted(registry))}."
        )
    return [registry[name] for name in script_args.reward_funcs]
