"""A synthetic response-token LM for exercising real sampling/gradients, not NLI quality.

Each response is a single vocabulary token. This makes tiny CPU GRPO tests useful
without teaching a random model XML syntax first. Production uses ordinary model
tokenizers and unconstrained generation; this vocabulary is never installed there.
"""
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from rlcr.data.dico_nli.labels import LABELS


def make_response_model():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    tokens = (
        ["[PAD]", "[UNK]", "[EOS]"]
        + [
            f"<answer>{label}</answer><confidence>{confidence}</confidence>"
            for label in LABELS
            for confidence in ("0.25", "0.75")
        ]
        + ["<answer>UNKNOWN</answer><confidence>0</confidence>"]
    )
    vocab = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        padding_side="left",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}"
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=512,
            pad_token_id=0,
            eos_token_id=2,
            attention_dropout=0.0,
        )
    ).eval()
    return model, tokenizer
