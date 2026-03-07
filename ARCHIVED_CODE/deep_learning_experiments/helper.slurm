#!/bin/bash
#SBATCH -J helper              # Job name
#SBATCH -o logs/helper.out          # Standard output and error log
#SBATCH -e logs/helper.err          # Standard error log
#SBATCH -p sched_mit_hill      # Partition (queue) name # mit_normal_gpu sched_mit_hill
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 4                   # Number of CPU cores
#SBATCH -t 4:00:00            # Runtime in HH:MM:SS (adjust as needed)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdong@mit.edu

# Load required modules (if any)
# module load python  # Adjust based on your Python version

# Activate your Conda environment
# source ~/.zshrc
# conda init zsh

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh # Adjust path if Conda is installed elsewhere
conda activate myenv


# Run your script (replace with your actual script)
python -u /home/pdong/Astro\ UROP/helper.py
