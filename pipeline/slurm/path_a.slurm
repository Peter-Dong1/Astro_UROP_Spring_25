#!/bin/bash
#SBATCH -J path_a_features
#SBATCH -o logs/path_a.out
#SBATCH -e logs/path_a.err
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 12:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdong@mit.edu

# PATH A — single-node feature extraction (produces all 11 features in one job)
# Run after: inaccessible_lightcurves.txt has been generated (python helper.py)
# Run before: slurm/analyze.slurm

PIPELINE_DIR="$SLURM_SUBMIT_DIR"

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate myenv

mkdir -p "$PIPELINE_DIR/logs"

echo "Pipeline dir: $PIPELINE_DIR"
echo "Starting PATH A feature extraction..."

python -u "$PIPELINE_DIR/run_feature_extraction.py"
