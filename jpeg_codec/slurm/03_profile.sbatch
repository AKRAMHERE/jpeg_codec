#!/bin/bash
# Function-level profile with gprof (ships with the GNU toolchain, no extra module needed).
#   sbatch slurm/03_profile.sbatch
# -fno-inline makes small helpers visible as separate rows; the absolute speed of this build is
# lower than the normal -O3 build, but the SHAPE of the profile (which stage dominates) is what you read.
#SBATCH --job-name=jlc-prof
#SBATCH --partition=dev_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000mb
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh

make BUILD=build_prof OPT="-O2 -pg -fno-inline" LDFLAGS=-pg
mkdir -p results
WORK="$(mktemp -d)"
( cd "${WORK}" && "${SLURM_SUBMIT_DIR}/build_prof/jlc_bench" --sizes 2048 --quality 50 --reps 5 --fused 0 > /dev/null )
gprof -b -p build_prof/jlc_bench "${WORK}/gmon.out" > results/gprof_flat.txt
gprof -b -q build_prof/jlc_bench "${WORK}/gmon.out" > results/gprof_callgraph.txt
head -25 results/gprof_flat.txt
