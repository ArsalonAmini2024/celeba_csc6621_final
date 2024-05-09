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

# Set up the path for the virtual environment
VENV_PATH=~/Desktop/celeba_csc6621_final/HPC/venv

# Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Upgrade pip and install the necessary packages
pip install --upgrade pip
pip install pandas scikit-learn tensorflow

# Run the Python training script within this environment
srun python3 ~/Desktop/celeba_csc6621_final/HPC/train_model_v1.py

# Optional: Deactivate the virtual environment
deactivate


