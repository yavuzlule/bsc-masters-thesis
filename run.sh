#!/bin/bash
#SBATCH --job-name=bsc_relish_pipeline_1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Load environment (adjust as needed)
module load python/3.10

# Go to project directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment
source .venv/bin/activate

# Ensure output directory exists
mkdir -p logs runs

# Run training
python -m bsc_relish.  \
  --config models/logistic_regression/config.yaml