import re
from .accuracy import accuracy_reward
from .format import format_reward


def brier_reward(format_pattern, completions, answer, source=None, **kwargs):
    """Reward function that checks if the completion is correct."""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = []
    correctness_rewards = accuracy_reward(format_pattern, completions, answer, source)
    format_rewards = format_reward(format_pattern, completions)
    for content, cr, fr in zip(completion_contents, correctness_rewards, format_rewards):
        if fr == 0:
            matches.append(0)
        else:
            # extract the confidence and give the reward as brier score
            confidence_matches = re.findall(
                confidence_pattern, content, re.DOTALL | re.MULTILINE
            )  # Get all <confidence>...</confidence> occurrences
            last_confidence = (
                confidence_matches[-1] if confidence_matches else ""
            )  # Get the last confidence, if exists
            if last_confidence == "":
                matches.append(0)
            else:
                try:
                    conf = float(last_confidence)
                    reward = 1 - (cr - conf) ** 2
                    matches.append(reward)
                except:
                    print(
                        "Could not parse confidence: ", last_confidence, "Something might be wrong"
                    )
                    matches.append(0)
    return matches
