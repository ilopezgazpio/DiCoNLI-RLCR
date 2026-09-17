from rlcr.text.prompts import get_sys_prompt
from .generation import make_generation_dataset


def process_dataset(dataset, script_args):
    if script_args.task_spec != "gen":
        raise ValueError("Only task_spec: gen is supported.")
    sys_prompt = get_sys_prompt(script_args.sys_prompt_name)
    return make_generation_dataset(dataset, sys_prompt)
