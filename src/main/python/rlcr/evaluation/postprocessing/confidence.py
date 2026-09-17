"""Confidence reprompting and the answer-token probability baseline."""
import re
import numpy as np
from rlcr.inference.generator import hf_generate


def add_answer_probability(tokenizer, outputs, config):
    invalid_count = 0
    # Get the logprob for everything inside <answer> </answer>
    for output in outputs:
        for i in range(config.n):
            picked = output.outputs[i]
            tokens = [tokenizer.decode([token_id]) for token_id in picked.token_ids]
            probs = [np.exp(logprob) for logprob in picked.token_logprobs]
            # find the 2nd last and last occurence of token
            answer_indices = [i for i, token in enumerate(tokens) if token == "answer"]

            # Get last and second last, if available
            end_index = answer_indices[-1] if len(answer_indices) >= 1 else None
            start_index = answer_indices[-2] if len(answer_indices) >= 2 else None
            if start_index == None or end_index == None or end_index - start_index >= 30:
                output.outputs[i].text = output.outputs[i].text + f"<confidence> 0.5 </confidence>"
                invalid_count += 1
            else:
                selected_probs = probs[start_index:end_index]
                selected_tokens = tokens[start_index:end_index]
                avg_prob = sum(selected_probs) / len(selected_probs)
                output.outputs[i].text = (
                    output.outputs[i].text + f"<confidence> {avg_prob} </confidence>"
                )

    print(
        f"Number of invalid confidence calls for {config.name}: {invalid_count/(config.n*len(outputs))}"
    )
    return invalid_count / (config.n * len(outputs))


def ensure_confidence(model, tokenizer, texts, outputs, config):
    inst = "Thinking time ended \n\n. My verbalized confidence in my answer as a number between 0 and 100 is equal to "
    prompts = []
    for text, output in zip(texts, outputs):
        for i in range(config.n):
            prompts.append(text + output.outputs[i].text + inst)

    verb_outputs = hf_generate(
        model,
        tokenizer,
        prompts,
        n=1,
        temperature=0,
        max_tokens=20,
        seed=config.seed,
        batch_size=config.hf_batch_size,
    )

    conf_calls_needed = 0
    counter = 0
    for output in outputs:
        for i in range(config.n):
            conf_pattern = r"<confidence>(.*?)</confidence>"
            conf_matches = re.findall(
                conf_pattern, output.outputs[i].text, re.DOTALL | re.MULTILINE
            )
            last_confidence = conf_matches[-1] if conf_matches else ""
            ## ONLY IF NO CONFIDENCE IS FOUND, USE THE CONFIDENCE FROM THE VERB_OUTPUTS
            if last_confidence == "":
                last_confidence = verb_outputs[counter].outputs[0].text
                output.outputs[i].text = (
                    output.outputs[i].text + "<confidence>" + last_confidence + "</confidence>"
                )
                conf_calls_needed += 1
            counter += 1
    print(
        f"Number of confidence calls needed for {config.name}: {conf_calls_needed/(config.n*len(outputs))}"
    )
    return conf_calls_needed / (config.n * len(outputs))
