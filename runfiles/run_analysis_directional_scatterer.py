import os
directory = os.path.dirname(os.path.realpath(__file__))
#import sys
#sys.path.insert(0, '/home/minseokhwan/')

import numpy as np
import eschallot.mie.simulate_particle as sim

import warnings
warnings.filterwarnings('ignore')

# Wavelength & angle ranges over which to compute the final efficiencies and phase function
lam_plot = np.linspace(400, 500, 101)
theta_plot = np.linspace(0, 180, 361)*np.pi/180 # 0: forward, 180: backward
phi_plot = np.array([0,90])*np.pi/180

## Define Cost Function
weight_Q = 0.1
weight_theta_fwd = 1.0
weight_theta_tgt = 1.0

r_max = 2500

#identifier = 'directional_scattering_theta30_lossless_SiO2_bulk_rmax' + str(r_max) + '_wQ' + str(weight_Q) + '_wTF1.0_wTT' + str(weight_theta_tgt)
#identifier = 'directional_scattering_theta30_lossy_TiO2_Sarkar_rmax' + str(r_max) + '_wQ' + str(weight_Q) + '_wTF1.0_wTT' + str(weight_theta_tgt)

#identifier = 'directional_scattering_theta30_lossless_SiO2_bulk_rmax' + str(r_max) + '_wQ' + str(weight_Q) + '_wTF0.0_wTT' + str(weight_theta_tgt)
identifier = 'directional_scattering_theta30_lossy_TiO2_Sarkar_rmax' + str(r_max) + '_wQ' + str(weight_Q) + '_wTF0.0_wTT' + str(weight_theta_tgt)

with np.load(directory + '/topopt_result_' + identifier + '.npz') as data:
    N_layer = data['N_layer']
    r = data['r'][:int(N_layer)]
    n = data['n'][:,:int(N_layer)+1]

Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
    S1_mpE, S1_mpM, S2_mpE, S2_mpM = sim.simulate(lam_plot, theta_plot, phi_plot, r, n)
    
np.savez(directory + '/analysis_result_' + identifier,
    Q_sca=Q_sca,
    Q_abs=Q_abs,
    Q_ext=Q_ext,
    p=p,
    diff_CS=diff_CS,
    t_El=t_El,
    t_Ml=t_Ml,
    Q_sca_mpE=Q_sca_mpE,
    Q_sca_mpM=Q_sca_mpM,
    S1_mpE=S1_mpE,
    S1_mpM=S1_mpM,
    S2_mpE=S2_mpE,
    S2_mpM=S2_mpM,
)

print(' ### Simulation Done', flush=True)