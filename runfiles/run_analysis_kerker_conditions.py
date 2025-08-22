import os
directory = os.path.dirname(os.path.realpath(__file__))
#import sys
#sys.path.insert(0, '/home/minseokhwan/')

import numpy as np
import eschallot.mie.simulate_particle as sim
import eschallot.util.read_mat_data as rmd

import warnings
warnings.filterwarnings('ignore')

# Wavelength & angle ranges over which to compute the final efficiencies and phase function
lam_plot = np.linspace(400, 700, 301)
theta_plot = np.linspace(0, 180, 361)*np.pi/180 # 0: forward, 180: backward
phi_plot = np.array([0,90])*np.pi/180

#identifier = 'kerker_conditions_fwd[450]_bwd[[450]]_lossy_Si_Schinke_rmax250_weights1000.0_0_100.0'
identifier = 'kerker_conditions_fwd[[450]]_bwd[450]_lossy_Si_Schinke_rmax250_weights0_1000.0_100.0'

with np.load(directory + '/Fig5/topopt_result_' + identifier + '.npz') as data:
    #idx = 0
    idx = 3
    N_layer = data['N_layer'][idx]
    r = data['r'][idx,:int(N_layer)]
    
    # Create n
    #mat_profile = np.array(['Air'] + ['Si_Schinke','Ag_palik']*2 + ['Si_Schinke'])
    mat_profile = np.array(['Air','Si_Schinke','SiO2_bulk','Ag_palik'])
    mat_type = list(set(mat_profile))
    raw_wavelength, mat_dict_plot = rmd.load_all(lam_plot, 'n_k', mat_type)
    
    n = np.zeros((np.size(lam_plot,0), np.size(mat_profile,0))).astype(complex)
    count = 0
    for mat in mat_profile:
        n[:,count] = mat_dict_plot[mat]
        count += 1

Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, Q_sca_mpE, Q_sca_mpM,\
    S1_mpE, S1_mpM, S2_mpE, S2_mpM = sim.simulate(lam_plot, theta_plot, phi_plot, r, n)
    
np.savez(directory + '/Fig5/analysis_result_' + identifier,
    r=r,
    n=n,
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