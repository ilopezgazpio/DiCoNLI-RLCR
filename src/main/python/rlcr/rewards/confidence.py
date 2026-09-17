import re


def mean_confidence_reward(completions, answer, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer]
    matches = []

    for content, e in zip(completion_contents, eval_contents):
        confidence_matches = re.findall(
            confidence_pattern, content, re.DOTALL | re.MULTILINE
        )  # Get all <confidence>...</confidence> occurrences
        last_confidence = (
            confidence_matches[-1] if confidence_matches else ""
        )  # Get the last confidence, if exists
        if last_confidence == "":
            matches.append(0.0)
        else:
            try:
                confidence = float(last_confidence)
                # clip confidence to be between 0 and 1
                confidence = max(0.0, min(confidence, 1.0))
            except:
                confidence = 0.0
            matches.append(confidence)
    return matches


def confidence_one_or_zero(completions, answer, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer]
    matches = []

    for content, e in zip(completion_contents, eval_contents):
        confidence_matches = re.findall(
            confidence_pattern, content, re.DOTALL | re.MULTILINE
        )  # Get all <confidence>...</confidence> occurrences
        last_confidence = (
            confidence_matches[-1] if confidence_matches else ""
        )  # Get the last confidence, if exists
        if last_confidence == "":
            matches.append(0.0)
        else:
            try:
                confidence = float(last_confidence)
                # clip confidence to be between 0 and 1
                confidence = max(0.0, min(confidence, 1.0))
            except:
                confidence = 0.0
            if abs(confidence - 1) < 0.01 or abs(confidence - 0) < 0.01:
                matches.append(1.0)
            else:
                matches.append(0.0)
    return matches
