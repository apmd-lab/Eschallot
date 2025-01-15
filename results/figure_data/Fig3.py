import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-20])

import numpy as np
import util.read_mat_data as rmd
import mie.simulate_particle as sim

lam = np.array([450])
theta = np.linspace(0, np.pi, 361)
phi = np.array([0,np.pi/2])

with np.load(directory + '/Fig3.npz') as data:
    r = data['r']
    
mat_profile = np.array(['Air'] + ['TiO2_Sarkar','SiO2_bulk']*3 + ['TiO2_Sarkar']) # Outer(including embedding medium) to inner
mat_type = list(set(mat_profile))
raw_wavelength, mat_dict = rmd.load_all(lam, 'n_k', mat_type)

n = np.zeros((lam.size, mat_profile.size)).astype(complex)
count = 0
for mat in mat_profile:
    n[:,count] = mat_dict[mat]
    count += 1

p = np.zeros((r.shape[0], r.shape[1], theta.size, phi.size))
p_norm = p.copy()
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        r_ij = r[i,j,:].copy()
        r_ij = r_ij[r_ij != 0]
        n_ij = n[:,:r_ij.size+1]
        
        _, _, _, p_ij, _, _, _, _, _, _, _, _, _ = sim.simulate(lam, theta, phi, r_ij, n_ij)
        
        p[i,j,:,:] = p_ij[0,:,:]
        
        # Peak-Finding Algorithm
        if i == 0:
            tgt_peak = 0
        elif i == r.shape[0]:
            tgt_peak = -1
        else:
            lshift = p_ij[0,1:-1,j] - p_ij[0,:-2,j]
            rshift = p_ij[0,1:-1,j] - p_ij[0,2:,j]
            lmask = np.where(lshift > 0, lshift*0 + 1, lshift*0)
            rmask = np.where(rshift > 0, lmask, rshift*0)
            all_peaks = np.nonzero(rmask)[0] + 1
            tgt_peak = all_peaks[np.argmin(np.abs(all_peaks/20 - i))]
        
        p_norm[i,j,:,:] = p_ij[0,:,:]/p_ij[0,tgt_peak,j]

np.savez(directory + '/Fig3', lam=lam, theta=theta, phi=phi, r=r, phase_fct=p, normalized_phase_fct=p_norm)