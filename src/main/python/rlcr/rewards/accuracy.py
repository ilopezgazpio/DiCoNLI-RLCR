import re
from math_verify import parse, verify
from rlcr.text.answers import exact_match_score
from .format import format_reward


def accuracy_reward(format_pattern, completions, answer, source=None, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    ans_pattern = r"<answer>(.*?)</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer]
    matches = []
    format_rewards = format_reward(format_pattern, completions)

    for content, e, fr in zip(completion_contents, eval_contents, format_rewards):
        if fr == 0:
            matches.append(0)
        else:
            ans_matches = re.findall(
                ans_pattern, content, re.DOTALL | re.MULTILINE
            )  # Get all <answer>...</answer> occurrences
            last_answer = ans_matches[-1] if ans_matches else ""  # Get the last answer, if exists
            # if source exists in key and is equal to hotpot, then use the exact match score
            if source is not None and source[0] == "hotpot":
                label = exact_match_score(last_answer, e)
            else:
                attempt = parse(last_answer)
                label = verify(e, attempt)
            matches.append(float(label))
    return matches
