"""These regression fixtures are intentionally CPU-only."""
import os

# Set before importing torch, including when tests run outside a sandbox.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM


@pytest.fixture
def tiny_model():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[EOS]",
                "question",
                "long",
                "answer",
                "yes",
                "no",
                "system",
                "user",
                "assistant",
                "test",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
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
            max_position_embeddings=128,
            pad_token_id=0,
            eos_token_id=2,
            attention_dropout=0.0,
        )
    ).eval()
    return model, tokenizer
