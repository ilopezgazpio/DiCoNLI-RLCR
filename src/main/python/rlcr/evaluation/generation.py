"""Generation and optional postprocessing for one evaluation configuration."""
from rlcr.inference.generator import hf_generate
from rlcr.models.inference import load_hf_generator
from rlcr.models.memory import clear_device_cache
from rlcr.models.tokenizer import load_tokenizer
from .postprocessing.answers import ensure_answers
from .postprocessing.classifier import classify_responses
from .postprocessing.confidence import add_answer_probability, ensure_confidence


def generate_columns(dataset, config):
    tokenizer = load_tokenizer(config.model)
    conversations = [example[config.tokenize_key] for example in dataset]
    prompt_ids = tokenizer.apply_chat_template(conversations, add_generation_prompt=True)
    texts = [tokenizer.decode(ids) for ids in prompt_ids]
    run_metrics = {}
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
            return_logprobs="confidence_prob" in config.tasks,
        )
        if "ans_at_end" in config.tasks:
            run_metrics["ans_calls_needed"] = ensure_answers(
                model, tokenizer, texts, outputs, config
            )
        if "confidence_prob" in config.tasks:
            run_metrics["invalid_confidence_prob_calls"] = add_answer_probability(
                tokenizer, outputs, config
            )
        if "confidence_at_end" in config.tasks:
            run_metrics["conf_calls_needed"] = ensure_confidence(
                model, tokenizer, texts, outputs, config
            )
    finally:
        del model
        clear_device_cache()

    class_scores = None
    if "gen_then_classify" in config.tasks:
        class_scores = classify_responses(dataset, outputs, config)
    columns = {
        f"{config.name}-output_{i}": [output.outputs[i].text for output in outputs]
        for i in range(config.n)
    }
    if class_scores is not None:
        columns[f"{config.name}-class_output"] = class_scores
    return columns, run_metrics
