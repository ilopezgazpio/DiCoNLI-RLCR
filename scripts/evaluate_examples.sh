#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root with its venv activated.

#HOTPOTQA Models
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/hotpot-models-commonsenseqa.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/hotpot-models-gpqa.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/hotpot-models-gsm8k.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/hotpot-models-hotpot-eval-em.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/hotpot-models-hotpot-vanilla-eval-em.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/hotpot-models-math-500.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/hotpot-models-simpleqa.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/hotpot-models-trivia.yaml

#MATH Models 
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/math-models-big-math-digits.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/math-models-commonsenseqa.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/math-models-gpqa.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/math-models-gsm8k.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/math-models-hotpot-vanilla-eval-em.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/math-models-math-500.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/math-models-simpleqa.yaml
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/math-models-trivia.yaml
