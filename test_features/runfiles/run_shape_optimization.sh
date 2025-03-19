#!/bin/bash

#SBATCH -o /home/gridsan/smin/python_scripts/Eschallot/slurm/run_shape_optimization.log-%j
#SBATCH -n 100

source /etc/profile
module load anaconda/2023a
export OMP_NUM_THREADS=1

mpirun -np 100 python /home/gridsan/smin/python_scripts/Eschallot/runfiles/run_shape_optimization.py
