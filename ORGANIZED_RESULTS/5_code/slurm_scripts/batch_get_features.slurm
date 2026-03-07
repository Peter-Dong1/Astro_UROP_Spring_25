#!/bin/bash
#SBATCH -J get_feature
#SBATCH -o logs/get_feature_%A_%a.out
#SBATCH -e logs/get_feature_%A_%a.err
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH -t 12:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdong@mit.edu

# Run only two array indices: 0 1
#SBATCH --array=0,100

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate myenv

python -u /home/pdong/Astro\ UROP/z\ New\ Feature\ Extraction\ Pipeline/run_feature_extraction.py \
    --job-id   $SLURM_ARRAY_TASK_ID \
    --num-jobs 200
