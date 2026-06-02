#!/bin/bash
#SBATCH -J path_b3_consolidate
#SBATCH -o logs/path_b3_consolidate.out
#SBATCH -e logs/path_b3_consolidate.err
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 12:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdong@mit.edu

# PATH B step 3 — merge all per-curve feature pickles into one DataFrame
# Run after:  ALL path_b_2_extract array tasks have completed
# Run before: slurm/analyze.slurm

PIPELINE_DIR="$SLURM_SUBMIT_DIR"

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate myenv

mkdir -p "$PIPELINE_DIR/logs"

echo "Pipeline dir: $PIPELINE_DIR"
echo "Starting PATH B step 3: consolidating per-curve feature pickles..."

python -u "$PIPELINE_DIR/consolidate_features.py"
