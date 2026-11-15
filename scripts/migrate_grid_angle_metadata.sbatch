#!/bin/bash
#SBATCH --job-name=spires-grid-angle
# Site-specific scheduler settings: uncomment and set for your cluster.
# #SBATCH --partition=<partition>
# #SBATCH --account=<account>
# #SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%1

set -euo pipefail

repository=${SPIRES_IO_REPO:?SPIRES_IO_REPO is required (path to a spires-io checkout)}
contract_repository=${SPIRES_CONTRACT_REPO:?SPIRES_CONTRACT_REPO is required (path to a spires-contract checkout)}
archive_root=${SPIRES_ARCHIVE_ROOT:?SPIRES_ARCHIVE_ROOT is required (path to the NOAA-20 VIIRS output tree)}
run_directory=${SPIRES_MIGRATION_RUN_DIRECTORY:?SPIRES_MIGRATION_RUN_DIRECTORY is required}
environment=${SPIRES_PYTHON_ENV:?SPIRES_PYTHON_ENV is required (path to the conda/venv prefix with spires installed)}
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
