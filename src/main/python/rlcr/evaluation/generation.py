"""Raw generation only: never repair answers or call a secondary judge."""
from rlcr.inference.generator import hf_generate
from rlcr.inference.prompts import render_prompt
from rlcr.models.inference import load_hf_generator
from rlcr.models.memory import clear_device_cache
from rlcr.models.tokenizer import load_tokenizer


def generate_columns(dataset, config):
    tokenizer = load_tokenizer(config.model)
    texts = [render_prompt(tokenizer, row[config.tokenize_key]) for row in dataset]
    model = load_hf_generator(config)
    try:
        outputs = hf_generate(
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
    columns = {
        f"{config.name}-output_{i}": [output.outputs[i].text for output in outputs]
        for i in range(config.n)
    }
    return columns, {"examples": len(outputs), "completions": len(outputs) * config.n}
