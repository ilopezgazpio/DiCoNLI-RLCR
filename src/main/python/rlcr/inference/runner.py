"""Ad-hoc inference through the same loading and generation code as evaluation."""
from rlcr.models.inference import load_hf_generator
from rlcr.models.memory import clear_device_cache
from rlcr.models.tokenizer import load_tokenizer
from rlcr.text.prompts import get_sys_prompt
from .generator import hf_generate


def run_inference(config, prompts, system_prompt_name):
    tokenizer = load_tokenizer(config.model)
    system_prompt = get_sys_prompt(system_prompt_name)
    conversations = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"\n\nPROBLEM: {prompt}\n\n"},
        ]
        for prompt in prompts
    ]
    ids = tokenizer.apply_chat_template(conversations, add_generation_prompt=True)
    texts = [tokenizer.decode(tokens) for tokens in ids]
    model = load_hf_generator(config)
    try:
        return hf_generate(
            model,
            tokenizer,
            texts,
            n=config.n,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
            batch_size=config.hf_batch_size,
        )
    finally:
        del model
        clear_device_cache()
