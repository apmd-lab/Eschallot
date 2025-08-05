#!/bin/bash

#SBATCH -o /home/minseokhwan/Eschallot/slurm/run_topology_optimization.log-%j
#SBATCH --partition=48core
#SBATCH --job-name=eschallot
#SBATCH --ntasks=12
##SBATCH --exclusive

export OMP_NUM_THREADS=1

/appl/intel/oneapi/mpi/2021.8.0/bin/mpirun -n 12 python /home/minseokhwan/Eschallot/runfiles/run_topology_optimization_directional_scatterer.py