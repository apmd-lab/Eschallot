#!/bin/bash

#SBATCH -o /home/minseokhwan/Eschallot/slurm/run_monte_carlo.log-%j
#SBATCH --partition=32core
#SBATCH --job-name=eschallot
#SBATCH --ntasks=64

export OMP_NUM_THREADS=1

/appl/intel/oneapi/mpi/2021.8.0/bin/mpirun -n 64 python /home/minseokhwan/Eschallot/runfiles/run_monte_carlo_directional_reflector.py