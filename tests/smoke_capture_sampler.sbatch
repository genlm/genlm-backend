#!/bin/bash
#SBATCH --partition=cuda13pr.24h
#SBATCH --gres=gpumem:80G,gpu:nvidia_rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --time=00:45:00
#SBATCH --job-name=capture-fix
#SBATCH --output=/cluster/scratch/cpasti/smoke-capture/%x.%j.out
#SBATCH --error=/cluster/scratch/cpasti/smoke-capture/%x.%j.err

set -euo pipefail
mkdir -p /cluster/scratch/cpasti/smoke-capture

source /cluster/home/cpasti/miniconda3/bin/activate genlm-latent
cd /cluster/scratch/cpasti/genlm-backend

export HF_HOME=/cluster/scratch/cpasti/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python tests/smoke_capture_sampler.py "$@"
