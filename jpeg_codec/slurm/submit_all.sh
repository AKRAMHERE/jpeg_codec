#!/bin/bash
# Convenience wrapper: submit the benchmark to both CPU partitions you want to compare.
#   bash slurm/submit_all.sh
# Run it from the project root on a bwUniCluster login node.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
for P in cpu_il cpu; do
    sbatch --partition="${P}" slurm/02_bench.sbatch
done
echo "Check progress with:  squeue --me    (do not run it in a loop; once every 30 s at most)"
