#!/bin/bash

#SBATCH -n 168

export OMP_NUM_THREADS=1

mpirun -np 168 python /filepath/run_monte_carlo.py
