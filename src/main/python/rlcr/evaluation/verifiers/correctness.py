import re
from math_verify import parse, verify
from rlcr.text.answers import exact_match_score


def gen_correctness_reward(completions, answer, **kwargs):
    """Reward function that checks if the answer is correct or not
    The answer must be present within the answer tags.
    For math datasets, the correctness is checked using huggingface math-verify.
    For factual datasets, the correctness is checked using exact match.

    """
    ans_pattern = r"<answer>(.*?)</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer]
    matches = []

    for content, e in zip(completion_contents, eval_contents):
        # Get all <answer>...</answer> occurrences
        ans_matches = re.findall(ans_pattern, content, re.DOTALL | re.MULTILINE)
        # Get the last answer, if exists
        last_answer = ans_matches[-1] if ans_matches else ""
        attempt = parse(last_answer)
        label = verify(e, attempt)
        if label == 0:
            label = exact_match_score(last_answer, e)
        matches.append(float(label))

    return matches
