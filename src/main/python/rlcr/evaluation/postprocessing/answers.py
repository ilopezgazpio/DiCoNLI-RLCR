"""Add an explicit final answer when the response lacks answer tags."""
import re
from rlcr.inference.generator import hf_generate


def ensure_answers(model, tokenizer, texts, outputs, config):
    inst = "Thinking time ended \n\n. My final answer is "
    prompts = []
    for text, output in zip(texts, outputs):
        for i in range(config.n):
            prompts.append(text + output.outputs[i].text + inst)
    ans_outputs = hf_generate(
        model,
        tokenizer,
        prompts,
        n=1,
        temperature=0,
        max_tokens=50,
        seed=config.seed,
        batch_size=config.hf_batch_size,
    )

    ans_calls_needed = 0
    counter = 0
    for out in outputs:
        for j in range(config.n):
            # first try to extract the answer from the output
            ans_pattern = r"<answer>(.*?)</answer>"
            ans_matches = re.findall(
                ans_pattern, out.outputs[j].text, re.DOTALL | re.MULTILINE
            )  # Get all <answer>...</answer> occurrences
            last_answer = ans_matches[-1] if ans_matches else ""  # Get the last answer, if exists
            ## ONLY IF NO ANSWER IS FOUND, USE THE ANSWER FROM THE ANS_OUTPUTS
            if last_answer == "":
                last_answer = ans_outputs[counter].outputs[0].text
                out.outputs[j].text = out.outputs[j].text + "<answer> " + last_answer + " </answer>"
                ans_calls_needed += 1
            counter += 1
    print(
        f"Number of answer calls needed for {config.name}: {ans_calls_needed/(config.n*len(outputs))}"
    )
    return ans_calls_needed / (config.n * len(outputs))
