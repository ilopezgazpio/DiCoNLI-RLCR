"""Ad-hoc inference through the same loading and generation code as evaluation."""
from rlcr.models.inference import load_hf_generator
from rlcr.models.memory import clear_device_cache
from rlcr.models.tokenizer import load_tokenizer
from .generator import hf_generate
from .prompts import render_prompt


def run_inference(config, prompts, system_prompt=None):
    tokenizer = load_tokenizer(config.model)
    conversations = [
        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + [{"role": "user", "content": prompt}]
        for prompt in prompts
    ]
    texts = [render_prompt(tokenizer, conversation) for conversation in conversations]
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
