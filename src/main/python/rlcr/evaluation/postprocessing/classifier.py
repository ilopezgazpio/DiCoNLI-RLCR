"""Score generated responses with an outcome/classifier model."""
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rlcr.models.memory import clear_device_cache


def classify_responses(dataset, outputs, config):
    key = "problem" if "problem" in dataset.column_names else "question"
    if config.split_at_confidence:
        for output in outputs:
            output.outputs[0].text = output.outputs[0].text.split("<confidence>")[0]
    texts = [
        f"\n\nPROBLEM: {dataset[i][key]}\n\nEND OF PROBLEM\n\n"
        f"MODEL'S RESPONSE: {output.outputs[0].text}\n\nEND OF RESPONSE\n\n"
        for i, output in enumerate(outputs)
    ]
    model = AutoModelForSequenceClassification.from_pretrained(
        config.class_model,
        device_map="auto",
        torch_dtype="auto"
        if config.torch_dtype in (None, "auto")
        else getattr(torch, config.torch_dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(config.class_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = model.get_input_embeddings().weight.device
    scores = []
    try:
        for start in tqdm(range(0, len(texts), config.hf_batch_size), desc="Classifying texts"):
            inputs = tokenizer(
                texts[start : start + config.hf_batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits.float()
                probabilities = (
                    logits[:, 0].sigmoid() if logits.shape[-1] == 1 else logits.softmax(-1)[:, 1]
                )
                scores.extend(probabilities.cpu().tolist())
    finally:
        del model
        clear_device_cache()
    return scores
