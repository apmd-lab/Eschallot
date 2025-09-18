import os
directory = os.path.dirname(os.path.realpath(__file__))
#import sys
#sys.path.insert(0, '/home/minseokhwan/')

import numpy as np
import eschallot.mie.simulate_particle as sim
import eschallot.util.read_mat_data as rmd
import time
import resource

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Nlayer', type=int, default=1)
parser.add_argument('--lmax', type=int, default=1)
args = parser.parse_args()

t1 = time.time()

# Wavelength & angle ranges over which to compute the final efficiencies and phase function
lam = np.linspace(400, 700, 301)
theta = np.linspace(0, 180, 361)*np.pi/180 # 0: forward, 180: backward
phi = np.array([0,90])*np.pi/180
    
# Create n
mat_list = np.array(['Air','Si_Schinke','SiO2_bulk'])
mat_type = list(set(mat_list))
raw_wavelength, mat_dict_plot = rmd.load_all(lam, 'n_k', mat_type)

r = np.linspace(1000, 10, 100)
r = r[:args.Nlayer]
if args.Nlayer % 2 == 0:
    mat_profile = np.array(['Air'] + int(args.Nlayer/2)*['Si_Schinke','SiO2_bulk'])
else:
    mat_profile = np.array(['Air'] + int((args.Nlayer - 1)/2)*['Si_Schinke','SiO2_bulk'] + ['Si_Schinke'])

n = np.zeros((np.size(lam,0), np.size(mat_profile,0))).astype(complex)
count = 0
for mat in mat_profile:
    n[:,count] = mat_dict_plot[mat]
    count += 1

Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
    S1_mpE, S1_mpM, S2_mpE, S2_mpM = sim.simulate(lam, theta, phi, r, n, lmax=args.lmax)

t2 = time.time()
max_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

np.savez(directory + '/time_memory_per_job', t=t2-t1, max_mem_kb=max_mem_kb)