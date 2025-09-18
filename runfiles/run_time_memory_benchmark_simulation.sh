#!/bin/bash

#SBATCH -o /home/minseokhwan/Eschallot/slurm/run_benchmark.log-%j
#SBATCH --partition=32core
#SBATCH --job-name=eschallot_benchmark
#SBATCH --ntasks=1

export OMP_NUM_THREADS=1

python /home/minseokhwan/Eschallot/runfiles/time_memory_benchmark_simulation.py "$@"