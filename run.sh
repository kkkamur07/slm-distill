#!/bin/bash
#SBATCH --job-name=score_distillation_sweep
#SBATCH --partition=mcml-dgx-a100-40x8,mcml-hgx-a100-80x4,mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting at: $(date)"

# Change to project directory
cd /dss/dsshome1/0C/ra85muk2/Desktop/Programming/score-distill

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

# Echo Training Started
echo "Starting the training of the score loss."

# Set CUDA devices (make all 4 visible)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the training script
# To run the sweep we need 
# python -m src.main -m --config sweep_config.yaml
python -m src.main

# Print completion time
echo "Finished at: $(date)"