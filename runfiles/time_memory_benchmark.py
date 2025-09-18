import os
directory = os.path.dirname(os.path.realpath(__file__))
#import sys
#sys.path.insert(0, '/home/minseokhwan/')

import numpy as np
import eschallot.mie.simulate_particle as sim
import eschallot.util.read_mat_data as rmd
import time
import subprocess

# (1) Layer number benchmark
t_nlayer = np.zeros((10, 10))
mem_nlayer = np.zeros((10, 10))

print('### Layer Number Benchmark', flush=True)
cnt = 0
for i in range(1, 100, 10):
    for j in range(10):
        if os.path.exists(directory + '/time_memory_per_job.npz'):
            os.remove(directory + '/time_memory_per_job.npz')
        
        proc = subprocess.Popen(["sbatch", directory + '/run_time_memory_benchmark_simulation.sh', '--Nlayer', str(i), '--lmax', '10'], stdout=subprocess.DEVNULL)
        while True:
            if os.path.exists(directory + '/time_memory_per_job.npz'):
                break
            time.sleep(1)
        
        with np.load(directory + '/time_memory_per_job.npz') as data:
            t_nlayer[cnt,j] = data['t']
            mem_nlayer[cnt,j] = data['max_mem_kb']/1024
        
        print('    Nlayer: ' + str(i) + ' | Try ' + str(j) + ' | ' + str(np.round(t_nlayer[cnt,j], 3)) + ' s | ' + str(np.round(mem_nlayer[cnt,j], 3)) + ' MB', flush=True)
    cnt += 1

# (2) lmax benchmark
t_lmax = np.zeros((10, 10))
mem_lmax = np.zeros((10, 10))

print('### lmax Number Benchmark', flush=True)
cnt = 0
for i in range(1, 100, 10):
    for j in range(10):
        if os.path.exists(directory + '/time_memory_per_job.npz'):
            os.remove(directory + '/time_memory_per_job.npz')
        
        proc = subprocess.Popen(["sbatch", directory + '/run_time_memory_benchmark_simulation.sh', '--Nlayer', '10', '--lmax', str(i)], stdout=subprocess.DEVNULL)
        while True:
            if os.path.exists(directory + '/time_memory_per_job.npz'):
                break
            time.sleep(1)
        
        with np.load(directory + '/time_memory_per_job.npz') as data:
            t_lmax[cnt,j] = data['t']
            mem_lmax[cnt,j] = data['max_mem_kb']/1024
        
        print('    lmax: ' + str(i) + ' Try ' + str(j) + ' | ' + str(np.round(t_nlayer[cnt,j], 3)) + ' s | ' + str(np.round(mem_nlayer[cnt,j], 3)) + ' MB', flush=True)
    cnt += 1
    
np.savez(directory + '/time_memory_benchmark_all',
    t_nlayer=t_nlayer,
    mem_nlayer=mem_nlayer,
    t_lmax=t_lmax,
    mem_lmax=mem_lmax,
)

print(' ### Benchmark Done', flush=True)