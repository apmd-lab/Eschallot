import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-20])

import numpy as np
import util.read_mat_data as rmd
import mie.simulate_particle as sim

lam = np.linspace(360, 830, 471)
theta = np.linspace(0, np.pi, 361)
phi = np.array([0,np.pi/2])

with np.load(directory + '/Fig1_Fig2.npz') as data:
    r = data['r']
    
mat_profile = np.array(['Air'] + ['TiO2_Sarkar','SiO2_bulk']*2) # Outer(including embedding medium) to inner
mat_type = list(set(mat_profile))
raw_wavelength, mat_dict = rmd.load_all(lam, 'n_k', mat_type)

n = np.zeros((lam.size, mat_profile.size)).astype(complex)
count = 0
for mat in mat_profile:
    n[:,count] = mat_dict[mat]
    count += 1

Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM, S1_mpE, S1_mpM, S2_mpE, S2_mpM = sim.simulate(lam, theta, phi, r, n)

np.savez(directory + '/Fig1_Fig2', lam=lam, theta=theta, phi=phi, r=r, n=n, Q_sca=Q_sca, phase_fct=p, mie_coeff_electric=t_El, mie_coeff_magnetic=t_Ml,
         Q_sca_electric=Q_sca_mpE, Q_sca_magnetic=Q_sca_mpM, S_perp_perp_electric=S1_mpE, S_perp_perp_magnetic=S1_mpM, S_par_par_electric=S2_mpE, S_par_par_magnetic=S2_mpM)