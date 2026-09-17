from transformers import AutoTokenizer


def load_tokenizer(model_name, **kwargs):
    kwargs = {"trust_remote_code": True, "padding_side": "left", **kwargs}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_training_tokenizer(model_name, model_kwargs):
    """Keep tokenizer revision and Hub access settings aligned with policy loading."""
    shared = {
        "revision",
        "trust_remote_code",
        "cache_dir",
        "local_files_only",
        "force_download",
        "proxies",
        "subfolder",
        "token",
        "use_auth_token",
    }
    kwargs = {key: value for key, value in (model_kwargs or {}).items() if key in shared}
    # Preserve Transformers' default for callers supplying an instantiated policy.
    kwargs.setdefault("trust_remote_code", False)
    return load_tokenizer(model_name, **kwargs)
