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


"""Adapt Transformers' data loader to buffered GRPO generation batches."""
import torch
import datasets
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available


def build_train_dataloader(trainer):
    if trainer.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = trainer.train_dataset
    data_collator = trainer.data_collator
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = trainer._remove_unused_columns(train_dataset, description="training")
    else:
        data_collator = trainer._get_collator_with_removed_columns(
            data_collator, description="training"
        )

    dataloader_params = {
        "batch_size": trainer._train_batch_size
        * trainer.args.steps_per_generation,  # < this is the change
        "collate_fn": data_collator,
        "num_workers": trainer.args.dataloader_num_workers,
        "pin_memory": trainer.args.dataloader_pin_memory,
        "persistent_workers": trainer.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = trainer._get_train_sampler()
        dataloader_params["drop_last"] = trainer.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker
        dataloader_params["prefetch_factor"] = trainer.args.dataloader_prefetch_factor

    return trainer.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
