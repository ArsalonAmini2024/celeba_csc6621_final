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

# Set up the conda environment name and path
CONDA_ENV_NAME="celeba_env"
CONDA_ENV_PATH=~/Desktop/celeba_csc6621_final/HPC/$CONDA_ENV_NAME

# Load the conda module (if your system uses modules)
# module load anaconda/2020.11  # Replace with the correct Anaconda module version

# Create the conda environment if it doesn't already exist
if ! conda info --envs | grep -q "$CONDA_ENV_NAME"; then
    conda create -y -p "$CONDA_ENV_PATH" python=3.8  # Adjust Python version as needed
fi

# Activate the conda environment
source activate "$CONDA_ENV_PATH"

# Install necessary packages via conda
conda install -y pandas scikit-learn tensorflow matplotlib seaborn

# Run the Python training script within this environment
srun python3 ~/Desktop/celeba_csc6621_final/HPC/train_model_v1.py

# Optional: Deactivate the conda environment
conda deactivate

