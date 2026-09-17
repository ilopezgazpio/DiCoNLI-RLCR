"""Evaluate generated answers with a separately loaded LLM judge."""
import re
from types import SimpleNamespace
from rlcr.inference.generator import hf_generate
from rlcr.models.inference import load_hf_generator
from rlcr.models.tokenizer import load_tokenizer
from rlcr.models.memory import clear_device_cache
from .reporting import build_report


def llm_confidence_verifier(
    local_dataset,
    config,
    judge_model="meta-llama/Llama-3.1-8B-Instruct",
    format_fn="confidence_format",
    judge_load_in_4bit=False,
    **kwargs,
):
    n = config.n
    # FIRST EXTRACT OUT ALL ANSWERS FROM THE MODEL OUTPUTS.
    extracted_answers = []
    for i in range(len(local_dataset)):
        q_spec_ans = []
        for j in range(n):
            pred = local_dataset[i][f"{config.name}-output_{j}"]
            ans_pattern = r"<answer>(.*?)</answer>"
            # Get all <answer>...</answer> occurrences
            ans_matches = re.findall(ans_pattern, pred, re.DOTALL | re.MULTILINE)
            # Get the last answer, if exists
            last_answer = ans_matches[-1] if ans_matches else ""
            if last_answer == "":
                last_answer = "I don't know"
            q_spec_ans.append(last_answer)
        extracted_answers.append(q_spec_ans)

    ####### DO LLM AS JUDGE SETUP #######
    sys_prompt = """
    You are a judge that will be given a question,ground truth answers and a model generated answer. There might be multiple ground truth answers. 
    The model generated answer is correct if it matches any of the ground truth answers.
    You will need to determine if the model generated answer is correct or not. 
    Your response should be a single word. 'YES' if the answer is correct and 'NO' if it is not.
    """

    prompts = []
    chosen_key = "question" if "question" in local_dataset.column_names else "problem"
    tokenizer = load_tokenizer(judge_model)

    # Generate prompts for each example
    for i in range(len(local_dataset)):
        for j in range(n):
            prompt = f"""
            Question: {local_dataset[i][chosen_key]}
            Ground Truth Answers: {local_dataset[i]["answer"]}
            Model Generated Answer: {extracted_answers[i][j]}
            """
            processed_prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
            tokenized_prompt = tokenizer.apply_chat_template(
                processed_prompt, truncation=False, add_generation_prompt=True
            )
            decoded_prompt = tokenizer.decode(tokenized_prompt)
            prompts.append(decoded_prompt)

    # Setup LLM and send prompts
    llm = load_hf_generator(
        SimpleNamespace(
            model=judge_model, torch_dtype=config.torch_dtype, load_in_4bit=judge_load_in_4bit
        )
    )
    outputs = hf_generate(
        llm,
        tokenizer,
        prompts,
        n=1,
        temperature=0,
        max_tokens=20,
        seed=config.seed,
        batch_size=config.hf_batch_size,
    )
    del llm
    clear_device_cache()

    ####### END OF LLM AS JUDGE SETUP #######

    ####### AGGREGATE RESPONSES #######

    responses = []
    for output in outputs:
        text_r = output.outputs[0].text
        if "yes" in text_r.lower():
            responses.append(1)
        else:
            responses.append(0)

    agg_responses = []
    # agg responses by taking groups of n and making a list of them
    for i in range(0, len(responses), n):
        agg_responses.append(responses[i : i + n])

    ####### END OF AGGREGATE RESPONSES #######

    return build_report(local_dataset, config, agg_responses)
