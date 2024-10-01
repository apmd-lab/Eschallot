#!/bin/bash

#SBATCH -o /home/gridsan/smin/python_scripts/Eschallot/slurm/run_topology_optimization.log-%j
#SBATCH -n 300

source /etc/profile
module load anaconda/2023a
export OMP_NUM_THREADS=1

mpirun -np 300 python /home/gridsan/smin/python_scripts/Eschallot/runfiles/run_topology_optimization.py
