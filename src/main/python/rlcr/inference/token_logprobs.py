from transformers import LogitsProcessor


class TokenLogprobs(LogitsProcessor):
    """Keep selected-token scores without retaining a vocabulary tensor per step."""

    def __init__(self):
        self.previous = None
        self.selected = []

    def record(self, token_ids):
        if self.previous is not None:
            values = self.previous.gather(1, token_ids[:, None]).squeeze(1)
            self.selected.append(values.cpu())
            self.previous = None

    def __call__(self, input_ids, scores):
        self.record(input_ids[:, -1])
        self.previous = scores.float().log_softmax(dim=-1)
        return scores
