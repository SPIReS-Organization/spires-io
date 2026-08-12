#!/bin/bash
#SBATCH --job-name=spires-grid-angle
#SBATCH --partition=blanca-snow
#SBATCH --account=blanca-snow
#SBATCH --qos=blanca-snow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%1

set -euo pipefail

repository=/projects/ropa5718/SPIReS-Organization/spires-io
contract_repository=/projects/ropa5718/SPIReS-Organization/spires-contract
archive_root=/scratch/alpine/ropa5718/spipy/output/noaa20-wy2023-2026/viirs/noaa20
run_directory=${SPIRES_MIGRATION_RUN_DIRECTORY:?SPIRES_MIGRATION_RUN_DIRECTORY is required}
environment=/projects/ropa5718/software/anaconda/envs/spipy14
tiles=(h08v04 h08v05 h09v04 h09v05 h10v04)
tile=${tiles[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$run_directory/logs" "$run_directory/manifests"
task_tmp="$run_directory/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$task_tmp"
export TMPDIR="$task_tmp"
export PYTHONPATH="$contract_repository:$repository${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=TRUE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$environment/bin/python" "$repository/scripts/migrate_grid_angle_metadata.py" \
    tile "$archive_root/$tile" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --manifest "$run_directory/manifests/${tile}.jsonl"
