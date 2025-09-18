import os
directory = os.path.dirname(os.path.realpath(__file__))
#import sys
#sys.path.insert(0, '/home/minseokhwan/')

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.integrate import trapezoid
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
theta_cost = np.linspace(0, 180, 181)*np.pi/180 # 0: forward, 180: backward
phi_cost = np.array([0,90])*np.pi/180

# Wavelength & angle ranges over which to compute the final efficiencies and phase function
lam_plot = np.array([450])
theta_plot = np.linspace(0, 180, 361)*np.pi/180 # 0: forward, 180: backward
phi_plot = np.array([0,90])*np.pi/180

class cost_obj:
    def __init__(self, phase_fct_tgt, ind_exclude):
        self.phase_fct_tgt = phase_fct_tgt
        self.ind_exclude = ind_exclude
    
    def cost(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml):
        cost = 1e2*np.mean((p[0,self.ind_exclude:,:] - self.phase_fct_tgt[self.ind_exclude:,:])**2)
        
        return cost
    
    def gradient(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml, r):
        jac = 2*(p[0,self.ind_exclude:,:] - self.phase_fct_tgt[self.ind_exclude:,:])[:,:,np.newaxis]*dp[0,self.ind_exclude:,:,:]/self.phase_fct_tgt.size
        jac = 1e2*np.sum(jac, axis=(0,1))
    
        return jac

## Define Cost Function
n_seed = 12
np.random.seed(n_seed)
pf_tgt = np.random.rand(theta_cost.size*phi_cost.size)
pf_tgt[0] *= 10
np.random.seed()
pf_tgt = gaussian_filter(pf_tgt, 10, mode='wrap', axes=0)
pf_tgt = np.vstack((pf_tgt[:theta_cost.size], np.flipud(pf_tgt[theta_cost.size:]))).T
pf_tgt[0,:] = pf_tgt[0,0]
pf_tgt[-1,:] = pf_tgt[-1,0]
pf_tgt = (np.tanh(8*(2*pf_tgt - 1)) + 1)/2
pf_norm = np.sum(np.pi*trapezoid(pf_tgt*np.sin(theta_cost)[:,np.newaxis], theta_cost, axis=0))
pf_tgt *= 0.3/pf_norm
np.savez(directory + '/phase_fct_tgt_seed_scale0_3' + str(n_seed), pf_tgt=pf_tgt)

custom_cost = cost_obj(pf_tgt, 16)

# Sweep Settings
r_min = 10
r_max = 3000
N_sweep = int(r_max - r_min) + 1
d_low = 5
max_layers = None

if 'Ag_palik' in mat_needle:
    output_filename = directory + '/topopt_result_arbitray_phase_fct_scale0_3_lossy_' + str(mat_profile[1]) + '_rmax' + str(r_max) + '_seed' + str(n_seed)
else:
    output_filename = directory + '/topopt_result_arbitray_phase_fct_scale0_3_lossless_' + str(mat_profile[1]) + '_rmax' + str(r_max) + '_seed' + str(n_seed)

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