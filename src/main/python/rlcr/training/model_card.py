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


"""Model-card export, separate from the optimization loop."""
import os
import textwrap
from typing import Optional, Union
from transformers import is_wandb_available
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

if is_wandb_available():
    import wandb


def create_model_card(
    trainer,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    tags: Union[str, list[str], None] = None,
):
    """Write the model card on the main worker only."""
    if not trainer.is_world_process_zero():
        return

    if hasattr(trainer.model.config, "_name_or_path") and not os.path.isdir(
        trainer.model.config._name_or_path
    ):
        base_model = trainer.model.config._name_or_path
    else:
        base_model = None

    tags = tags or []
    if isinstance(tags, str):
        tags = [tags]

    if hasattr(trainer.model.config, "unsloth_version"):
        tags.append("unsloth")

    citation = textwrap.dedent(
        """\
        @article{zhihong2024deepseekmath,
            title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
            author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
            year         = 2024,
            eprint       = {arXiv:2402.03300},
        }
        """
    )

    model_card = generate_model_card(
        base_model=base_model,
        model_name=model_name,
        hub_model_id=trainer.hub_model_id,
        dataset_name=dataset_name,
        tags=tags,
        wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
        comet_url=get_comet_experiment_url(),
        trainer_name="GRPO",
        trainer_citation=citation,
        paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        paper_id="2402.03300",
    )

    model_card.save(os.path.join(trainer.args.output_dir, "README.md"))
