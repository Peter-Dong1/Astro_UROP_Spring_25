#!/bin/bash
#SBATCH -J path_b1_split
#SBATCH -o logs/path_b1_split.out
#SBATCH -e logs/path_b1_split.err
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 1:00:00
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdong@mit.edu

# PATH B step 1 — split FITS file paths into 28 partition pickles
# (saves path strings only, NOT loaded DataFrames — memory is ~4 GB vs old 350 GB)
# Run after:  inaccessible_lightcurves.txt has been generated (python helper.py)
# Run before: slurm/path_b_2_extract.slurm

PIPELINE_DIR="$SLURM_SUBMIT_DIR"

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate myenv

mkdir -p "$PIPELINE_DIR/logs"

echo "Pipeline dir: $PIPELINE_DIR"
echo "Starting PATH B step 1: splitting file paths into partitions..."

python -u "$PIPELINE_DIR/split_curves.py"
