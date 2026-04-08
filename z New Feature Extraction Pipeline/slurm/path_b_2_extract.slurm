#!/bin/bash
#SBATCH -J path_b2_extract
#SBATCH -o logs/path_b2_extract_%A_%a.out
#SBATCH -e logs/path_b2_extract_%A_%a.err
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH --cpus-per-task=85
#SBATCH -t 48:00:00
#SBATCH --mem=128G
#SBATCH --array=0-28
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdong@mit.edu

# PATH B step 2 — extract features for one partition (SLURM array job)
# Each of the 29 tasks (0-28) processes one partition file independently.
# Run after:  slurm/path_b_1_split.slurm has completed
# Run before: slurm/path_b_3_consolidate.slurm

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate myenv

mkdir -p "$PIPELINE_DIR/logs"
mkdir -p "$PIPELINE_DIR/extracted_features"

IDX=${SLURM_ARRAY_TASK_ID}
CHUNK_ID=$(printf "%02d" $IDX)
CHUNK_FILE="$PIPELINE_DIR/data/split_light_curves/light_curves_partition_${CHUNK_ID}.pkl"
OUTPUT_DIR="$PIPELINE_DIR/extracted_features"

echo "Pipeline dir: $PIPELINE_DIR"
echo "Processing partition ${CHUNK_ID} → output: $OUTPUT_DIR"

python -u "$PIPELINE_DIR/batch_feature_extraction.py" \
    --chunk-file "$CHUNK_FILE" \
    --chunk-id   "$IDX" \
    --output-dir "$OUTPUT_DIR"
