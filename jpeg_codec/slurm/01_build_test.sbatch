#!/bin/bash
# Build the library on a compute node and run the unit tests.
# Uses the short development queue (30 min limit, one running job at a time).
#   sbatch slurm/01_build_test.sbatch
#SBATCH --job-name=jlc-build
#SBATCH --partition=dev_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4000mb
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh

make clean
make -j "${SLURM_CPUS_PER_TASK:-1}"
make test
echo "build + tests OK on $(hostname)"
