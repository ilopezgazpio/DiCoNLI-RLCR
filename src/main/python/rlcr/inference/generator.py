import torch
from tqdm import tqdm
from .completion import Completion
from .generation import Generation
from .token_logprobs import TokenLogprobs


def hf_generate(
    model, tokenizer, prompts, n, temperature, max_tokens, seed, batch_size, return_logprobs=False
):
    if n < 1 or batch_size < 1:
        raise ValueError("n and batch_size must be positive.")
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    do_sample = temperature is not None and temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "num_beams": 1,
        "num_return_sequences": n if do_sample else 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": False,
        "output_scores": False,
        "output_logits": False,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    device = model.get_input_embeddings().weight.device
    all_outputs = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="HF generation"):
        batch_prompts = prompts[start : start + batch_size]
        # Greedy decoding returns one sequence; duplicate its result for n > 1.
        num_sequences = n if do_sample else 1
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        collector = TokenLogprobs() if return_logprobs else None
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                **generation_kwargs,
                logits_processor=[collector] if collector is not None else None,
            )
        completion_ids = generated[:, prompt_len:]
        token_logprobs = None
        if collector is not None:
            collector.record(generated[:, -1])
            token_logprobs = torch.stack(collector.selected, dim=1).tolist()
            del collector

        eos_ids = generation_kwargs["eos_token_id"]
        eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
        completions = []
        for row, ids in enumerate(completion_ids.tolist()):
            end = next((i + 1 for i, token in enumerate(ids) if token in eos_ids), len(ids))
            ids = ids[:end]
            completions.append(
                Completion(
                    text=tokenizer.decode(ids, skip_special_tokens=True),
                    token_ids=ids,
                    token_logprobs=token_logprobs[row][:end]
                    if token_logprobs is not None
                    else None,
                )
            )
        for i in range(len(batch_prompts)):
            group = completions[i * num_sequences : (i + 1) * num_sequences]
            if not do_sample:
                completion = group[0]
                group = [
                    Completion(
                        completion.text,
                        completion.token_ids.copy(),
                        completion.token_logprobs.copy() if return_logprobs else None,
                    )
                    for _ in range(n)
                ]
            all_outputs.append(Generation(outputs=group))
    return all_outputs
