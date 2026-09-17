import re


def format_reward(format_pattern, completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    if format_pattern == "tbac":
        pattern = r".*?</think>\s*<analysis>.*?</analysis>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    elif format_pattern == "ta":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*\Z"
    elif format_pattern == "tac":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    elif format_pattern == "tabc":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>.*?</confidence>\s*\Z"
    confidence_pattern = r"<confidence>(.*?)</confidence>"

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents
    ]
    matches = [1.0 if match else 0.0 for match in matches]

    # if it matches, check if the confidence is between 0 and 1
    for i, match in enumerate(matches):
        if match:
            content = completion_contents[i]
            if "c" in format_pattern:
                confidence_matches = re.findall(
                    confidence_pattern, content, re.DOTALL | re.MULTILINE
                )  # Get all <confidence>...</confidence> occurrences
                last_confidence = (
                    confidence_matches[-1] if confidence_matches else ""
                )  # Get the last confidence, if exists
                if last_confidence == "":
                    matches[i] = 0.0
                else:
                    try:
                        confidence = float(last_confidence)
                        if confidence < 0 or confidence > 1:
                            matches[i] = 0.0
                        else:
                            matches[i] = 1

                    except:
                        matches[i] = 0.0
    return matches
