"""Select reusable rewards or their explicit four-label DiCo-NLI bindings."""
from .accuracy import accuracy_reward
from .brier import brier_reward
from .format import format_reward
from .dico_nli import dico_accuracy_reward, dico_brier_reward, dico_format_reward

LABEL_REWARDS = {"accuracy", "brier", "dico_accuracy", "dico_brier"}
DICO_REWARDS = {"dico_accuracy", "dico_brier", "dico_format"}


def build_reward_functions(script_args):
    registry = {
        "format": format_reward,
        "accuracy": accuracy_reward,
        "brier": brier_reward,
        "dico_accuracy": dico_accuracy_reward,
        "dico_brier": dico_brier_reward,
        "dico_format": dico_format_reward,
    }
    unknown = set(script_args.reward_funcs) - registry.keys()
    if unknown:
        raise ValueError(
            f"Unknown reward functions: {', '.join(sorted(unknown))}. "
            f"Supported: {', '.join(sorted(registry))}."
        )
    selected = set(script_args.reward_funcs)
    if selected & DICO_REWARDS and selected - DICO_REWARDS:
        raise ValueError("Do not mix unrestricted and DiCo-NLI rewards in one run.")
    return [registry[name] for name in script_args.reward_funcs]
