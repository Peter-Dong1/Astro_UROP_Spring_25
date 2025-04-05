#!/bin/bash
#SBATCH -J testing              # Job name
#SBATCH -o logs/testing.out          # Standard output and error log
#SBATCH -e logs/testing.err          # Standard error log
#SBATCH -p mit_normal_gpu      # Partition (queue) name # mit_normal_gpu sched_mit_hill
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 4                   # Number of CPU cores
#SBATCH -t 12:00:00            # Runtime in HH:MM:SS (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdong@mit.edu
# Load required modules (if any)
# module load python  # Adjust based on your Python version

# Activate your Conda environment
# source ~/.zshrc
# conda init zsh

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh # Adjust path if Conda is installed elsewhere
conda activate myenv


# Run your script
python /home/pdong/Astro\ UROP/test_train.py
