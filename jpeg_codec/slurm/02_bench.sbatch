#!/bin/bash
# Per-stage encode/decode benchmark on ONE core of ONE node of the partition you submit to.
#   sbatch --partition=cpu_il slurm/02_bench.sbatch     # Intel Xeon Platinum 8358 (Ice Lake)
#   sbatch --partition=cpu    slurm/02_bench.sbatch     # AMD EPYC 9454
#   NATIVE=1 sbatch --partition=cpu_il slurm/02_bench.sbatch   # also rebuild with -march=native
# Add --exclusive to the sbatch line for the cleanest numbers (you then wait longer in the queue).
#SBATCH --job-name=jlc-bench
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000mb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh

PART="${SLURM_JOB_PARTITION:-local}"
BIN=build
TAG="${PART}"
if [[ "${NATIVE:-0}" == "1" ]]; then
    BIN="build_native_${PART}"
    TAG="${PART}_native"
    make BUILD="${BIN}" OPT="-O3 -march=native"
elif [[ ! -x build/jlc_bench ]]; then
    make
fi

mkdir -p results
# --- record exactly what the numbers were measured on
{
    echo "date:       $(date -Is)"
    echo "host:       $(hostname)"
    echo "partition:  ${PART}"
    echo "job id:     ${SLURM_JOB_ID:-none}"
    echo "compiler:   $(g++ --version | head -1)"
    echo "binary dir: ${BIN}"
    lscpu | grep -E 'Model name|Socket|Thread|Core|CPU MHz|L1d|L2|L3' || true
} > "results/meta_${TAG}.txt"
cat "results/meta_${TAG}.txt"

# --- synthetic images of growing size; staged (fused=0) and fused (fused=1) execution
for FUSED in 0 1; do
    srun --cpu-bind=cores "${BIN}/jlc_bench" \
        --sizes 512,1024,2048,4096 --quality 10,50,90 --reps 7 \
        --fused "${FUSED}" --label synthetic \
        --csv "results/bench_${TAG}_fused${FUSED}.csv"
done

# --- real photographs (512x512) at the same qualities
for IMG in camera astronaut texture; do
    if [[ -f "data/${IMG}.pgm" ]]; then
        srun --cpu-bind=cores "${BIN}/jlc_bench" --image "data/${IMG}.pgm" \
            --quality 10,50,90 --reps 15 --fused 0 \
            --csv "results/bench_${TAG}_${IMG}.csv" > /dev/null
    fi
done
echo "done: results/bench_${TAG}_*.csv"
