#!/bin/bash
#SBATCH --job-name=Amini-Arsalon-Celeba-image-classification-ResNet-random-weights
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=dgx

# Upgrade pip and install the necessary packages globally (or for the user only)
pip install --upgrade --user pip
pip install --user pandas scikit-learn tensorflow matplotlib seaborn tensorflow-addons

# Run the Python training script
srun python3 ~/Desktop/celeba_csc6621_final/HPC/train_model_v2.py



