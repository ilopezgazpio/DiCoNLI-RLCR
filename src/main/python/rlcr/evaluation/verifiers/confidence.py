import re


def confidence_extractor(response, **kwargs):
    """Extracts the confidence from the completions
    If a float is found within confidence tags, it is processed as follows:
    If the float is between 0 and 1, it is returned as is.
    If the float is between 1 and 100, it is divided by 100 and returned.
    If float is not directly found, the first number in the string is extracted and processed as above.
    If no float is found, 0 is returned.
    """
    conf_pattern = r"<confidence>(.*?)</confidence>"
    # Get all <confidence>...</confidence> occurrences
    conf_matches = re.findall(conf_pattern, response, re.DOTALL | re.MULTILINE)
    # Get the last confidence, if exists
    last_confidence = conf_matches[-1] if conf_matches else ""
    if last_confidence == "":
        return 0, 0.0
    else:
        try:
            confidence = float(last_confidence)
            if confidence > 1 and confidence <= 100:
                return 1, confidence / 100
            elif confidence >= 0 and confidence <= 1:
                return 1, confidence
            else:
                return 0, 0.0
        except:
            # extract the first number in the string
            first_number = re.search(r"-?\d+(?:\.\d+)?", last_confidence)
            if first_number:
                first_number = float(first_number.group())
                if first_number >= 0 and first_number <= 1:
                    return 1, first_number
                elif first_number > 1 and first_number <= 100:
                    return 1, first_number / 100
                else:
                    return 0, 0.0
            else:
                return 0, 0.0
