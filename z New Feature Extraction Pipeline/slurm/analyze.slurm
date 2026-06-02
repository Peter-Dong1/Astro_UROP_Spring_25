#!/bin/bash
#SBATCH -J analyze_features
#SBATCH -o logs/analyze.out
#SBATCH -e logs/analyze.err
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 12:00:00
#SBATCH --mem=180G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdong@mit.edu

# Final analysis step (used by both PATH A and PATH B)
# Loads FEATURES_FILE, runs outlier detection, HDBSCAN clustering,
# UMAP, and generates all plots/CSVs.
# Run after:  slurm/path_a.slurm  OR  slurm/path_b_3_consolidate.slurm

PIPELINE_DIR="$SLURM_SUBMIT_DIR"

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate myenv

mkdir -p "$PIPELINE_DIR/logs"

echo "Pipeline dir: $PIPELINE_DIR"
echo "Starting analysis pipeline (outlier detection, HDBSCAN, UMAP, plots)..."

python -u "$PIPELINE_DIR/run_pipeline_on_features.py"
