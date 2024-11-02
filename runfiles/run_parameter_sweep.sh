#!/bin/bash

#SBATCH -o /home/gridsan/smin/python_scripts/Eschallot/slurm/run_parameter_sweep.log-%j
#SBATCH -n 1

source /etc/profile
module load anaconda/2023a
export OMP_NUM_THREADS=1

python /home/gridsan/smin/python_scripts/Eschallot/runfiles/run_parameter_sweep.py
