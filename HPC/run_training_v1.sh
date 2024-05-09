#!/bin/bash
#SBATCH --job-name=Amini-Arsalon-Celeba-image-classification-ResNet-random-weights
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:8
#SBATCH --partition=dgx

# Run the Python training script
srun python3 ~/Desktop/celeba_csc6621_final/HPC/train_model_v1.py
