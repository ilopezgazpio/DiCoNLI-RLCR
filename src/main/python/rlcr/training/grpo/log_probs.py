# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-bounded policy log probabilities for the GRPO objective."""

import torch
from trl.trainer.utils import selective_log_softmax


def per_token_log_probs(model, input_ids, attention_mask, logits_to_keep, temperature, batch_size):
    all_logps = []
    for start in range(0, input_ids.size(0), batch_size):
        ids = input_ids[start : start + batch_size]
        mask = attention_mask[start : start + batch_size]
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=logits_to_keep + 1).logits
        # Transformers 4.48 models may ignore logits_to_keep, so slice explicitly.
        logits = logits[:, :-1, :][:, -logits_to_keep:, :] / temperature
        all_logps.append(selective_log_softmax(logits, ids[:, -logits_to_keep:]))
    return torch.cat(all_logps, dim=0)
