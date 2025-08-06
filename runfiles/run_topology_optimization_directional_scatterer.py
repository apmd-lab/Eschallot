import os
directory = os.path.dirname(os.path.realpath(__file__))
#import sys
#sys.path.insert(0, '/home/minseokhwan/')

import numpy as np
import eschallot.optimization.topology_optimization as topopt
import time
import resource
from mpi4py import MPI
comm = MPI.COMM_WORLD

import warnings
warnings.filterwarnings('ignore')

# Materials
mat_profile = np.array(['Air','TiO2_Sarkar']) # Outer(including embedding medium) to inner
mat_needle = np.array(['TiO2_Sarkar','SiO2_bulk','Si_Schinke','Ag_palik']) # 'TiO2_Sarkar','Au_JC','Si_Schinke','Ag_palik', etc.

# Wavelength & angle ranges over which the cost function is defined
lam_cost = np.array([450])
theta_cost = np.array([0,70])*np.pi/180 # 0: forward, 180: backward
phi_cost = np.array([0])*np.pi/180

# Wavelength & angle ranges over which to compute the final efficiencies and phase function
lam_plot = np.array([450])
theta_plot = np.linspace(0, 180, 361)*np.pi/180 # 0: forward, 180: backward
phi_plot = np.array([0,90])*np.pi/180

class cost_obj:
    def __init__(self, weight_Q, weight_theta_fwd, weight_theta_tgt, Q_sca_threshold, logistic_beta):
        self.weight_Q = weight_Q
        self.weight_theta_fwd = weight_theta_fwd
        self.weight_theta_tgt = weight_theta_tgt
        self.Q_sca_threshold = Q_sca_threshold
        self.logistic_beta = logistic_beta
    
    def cost(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml):
        cost_Q_sca = -self.weight_Q*Q_sca[0]/(1 + np.exp(self.logistic_beta*(Q_sca[0] - self.Q_sca_threshold)))
        cost_p = self.weight_theta_fwd*p[0,0,0] - self.weight_theta_tgt*p[0,1,0]
        
        cost = cost_Q_sca + cost_p
        
        return cost
    
    def gradient(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml, r):
        logistic_fct = 1/(1 + np.exp(self.logistic_beta*(Q_sca[0] - self.Q_sca_threshold)))
        jac_Q_sca = -self.weight_Q*logistic_fct*(1 - self.logistic_beta*Q_sca[0]*np.exp(self.logistic_beta*(Q_sca[0] - self.Q_sca_threshold))*logistic_fct)*dQ_sca[0,:]
        jac_p = self.weight_theta_fwd*dp[0,0,0,:] - self.weight_theta_tgt*dp[0,1,0,:]
        
        jac = jac_Q_sca + jac_p
    
        return jac

## Define Cost Function
weight_Q = 0.1
weight_theta_fwd = 0.0
weight_theta_tgt = 1.0

custom_cost = cost_obj(
    weight_Q=weight_Q,
    weight_theta_fwd=weight_theta_fwd,
    weight_theta_tgt=weight_theta_tgt,
    Q_sca_threshold=0.2,
    logistic_beta=10,
    )

# Sweep Settings
r_min = 10
r_max = 2500
N_sweep = int(r_max - r_min) + 1
d_low = 5
max_layers = None

if 'Ag_palik' in mat_needle:
    output_filename = directory + '/topopt_result_directional_scattering_theta' + str(int(np.round(theta_cost[1]*180/np.pi))) + '_phi' + str(int(np.round(phi_cost[0]*180/np.pi))) + '_lossy_' + str(mat_profile[1]) + '_rmax' + str(r_max) + '_wQ' + str(weight_Q) + '_wTF' + str(weight_theta_fwd) + '_wTT' + str(weight_theta_tgt)
else:
    output_filename = directory + '/topopt_result_directional_scattering_theta' + str(int(np.round(theta_cost[1]*180/np.pi))) + '_phi' + str(int(np.round(phi_cost[0]*180/np.pi))) + '_lossless_' + str(mat_profile[1]) + '_rmax' + str(r_max) + '_wQ' + str(weight_Q) + '_wTF' + str(weight_theta_fwd) + '_wTT' + str(weight_theta_tgt)

t1 = time.time()
topopt.radius_sweep(output_filename,
                    r_min,
                    r_max,
                    N_sweep,
                    d_low,
                    max_layers,
                    mat_profile,
                    mat_needle,
                    lam_cost,
                    theta_cost,
                    phi_cost,
                    lam_plot,
                    theta_plot,
                    phi_plot,
                    custom_cost,
                    mat_data_dir=None,
                    lmax=None,
                    )
t2 = time.time()

max_mem_kb_proc = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
max_mem_kb = comm.reduce(max_mem_kb_proc, op=MPI.SUM, root=0)

if comm.rank == 0:
    print('### Time Elapsed: ' + str(np.round(t2 - t1, 3)) + ' s', flush=True)
    print('### Maximum Memory Usage: ' +str(np.round(max_mem_kb/1024**2, 3)) + ' GB', flush=True)