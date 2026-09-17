#!/usr/bin/env bash
#SBATCH --job-name=rlcr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

# Single-node template: adjust partition/account/time/GPU requests for your site.
# Submit from the repository root:
# sbatch scripts/slurm/train.sbatch /path/to/training.yaml
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch scripts/slurm/train.sbatch TRAIN_CONFIG [TRAIN_OVERRIDES...]" >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
rlcr_python="${RLCR_PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
rlcr_processes="${SLURM_GPUS_ON_NODE:-4}"
rlcr_config="$1"
shift

# One Slurm task launches one Accelerate worker per allocated GPU.
# Do not also request one Slurm task per GPU: that would launch duplicate workers.
srun --ntasks=1 "$rlcr_python" -m accelerate.commands.launch \
    --config_file configs/accelerate/zero2.yaml \
    --num_processes "$rlcr_processes" \
    --module rlcr train --config "$rlcr_config" "$@"
