#!/bin/bash

#SBATCH -n 48

source /etc/profile
module load anaconda/2023a
export OMP_NUM_THREADS=1

mpirun -np 48 python /home/gridsan/smin/python_scripts/MCLSOpt/MCLSOpt/runfiles/run_monte_carlo_parameter_sweep.py
