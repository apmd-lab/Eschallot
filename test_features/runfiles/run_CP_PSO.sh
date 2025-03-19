#!/bin/bash

#SBATCH -o CP_PSO.log-%j
#SBATCH -n 1

export OMP_NUM_THREADS=1

python /filepath/run_CP_PSO.py
