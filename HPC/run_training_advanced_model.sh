#!/bin/bash
#SBATCH --job-name=Amini-Arsalon-Celeba-image-classification-Resnet-50
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:8
#SBATCH --partition=dgx

module load cuda/10.1  # Load necessary modules
module load tensorflow/2.3.0

srun python3 path_to_your_script.py
