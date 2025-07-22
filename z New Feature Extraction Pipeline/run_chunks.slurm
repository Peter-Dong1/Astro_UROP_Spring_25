#!/bin/bash
#SBATCH -J extract_features
#SBATCH -o logs/extract_features_%A_%a.out
#SBATCH -e logs/extract_features_%A_%a.err
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH --cpus-per-task=85
#SBATCH -t 48:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdong@mit.edu

#SBATCH --array=0-11  # Run 12 jobs: array indices 0 through 11

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate myenv

# Set paths
IDX=${SLURM_ARRAY_TASK_ID}
CHUNK_ID=$(printf "%02d" $IDX)
CHUNK_FILE="/home/pdong/Astro UROP/z New Feature Extraction Pipeline/data/split_light_curves/light_curves_partition_${CHUNK_ID}.pkl"
OUTPUT_DIR="/home/pdong/Astro UROP/z New Feature Extraction Pipeline/extracted_features"
SCRIPT_PATH="/home/pdong/Astro UROP/z New Feature Extraction Pipeline/batch_feature_extraction.py"

# Run feature extraction for the current chunk
python -u "$SCRIPT_PATH" \
    --chunk-file "$CHUNK_FILE" \
    --chunk-id "$IDX" \
    --output-dir "$OUTPUT_DIR"
