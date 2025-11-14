#!/bin/bash
#SBATCH -J optimise             # Job name
#SBATCH -p gpu                   # GPU partition
#SBATCH -N 1                     # Run on a single node
#SBATCH --gres=gpu:1             # Request 2 GPUs
#SBATCH --constraint=a100-80gb 
#SBATCH --cpus-per-task=8        # 8 CPU cores (for data loading, etc.)
#SBATCH --mem=480G               # Total memory available to the job
#SBATCH -t 3-00:00               # Max runtime (24 hours)
#SBATCH -o logs/%x_%j.out        # STDOUT (%x=jobname, %j=jobid)
#SBATCH -e logs/%x_%j.err        # STDERR
#SBATCH --hint=multithread       # (optional) allow hyperthreading for dataloaders

# Load modules
module load python

# Activate your virtual environment
source /mnt/home/mliu1/env/bin/activate

# Print debug info
echo "Running on node $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run your training script
free -h
srun python star_flow.py
