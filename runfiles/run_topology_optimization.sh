#!/bin/bash

#SBATCH -o /home/minseokhwan/Eschallot/slurm/run_topology_optimization.log-%j
#SBATCH --partition=32core
#SBATCH --job-name=eschallot
#SBATCH --ntasks=8
##SBATCH --exclusive

export OMP_NUM_THREADS=1

##/appl/intel/oneapi/mpi/2021.8.0/bin/mpirun -n 8 python /home/minseokhwan/Eschallot/runfiles/run_topology_optimization_directional_scatterer.py
/appl/intel/oneapi/mpi/2021.8.0/bin/mpirun -n 8 python /home/minseokhwan/Eschallot/runfiles/run_topology_optimization_kerker_conditions.py