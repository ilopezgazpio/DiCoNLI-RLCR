"""Render prepared text or chat messages with no task-specific instructions."""


def render_prompt(tokenizer, prompt):
    if isinstance(prompt, str):
        return prompt
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
