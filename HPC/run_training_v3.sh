#!/bin/bash
#SBATCH --job-name=Amini-Arsalon-Celeba-image-classification-ResNet-random-weights
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:8
#SBATCH --partition=dgx

# Run the Python training script
srun python3 ~/Desktop/celeba_csc6621_final/HPC/train_model_v3.py