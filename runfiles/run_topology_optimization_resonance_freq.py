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
mat_profile = np.array(['Air','Si_Schinke']) # Outer(including embedding medium) to inner
mat_needle = np.array(['TiO2_Sarkar','SiO2_bulk','Au_JC','Si_Schinke','Ag_palik']) # 'TiO2_Sarkar','Au_JC','Si_Schinke','Ag_palik', etc.

# Wavelength & angle ranges over which the cost function is defined
lam_cost = np.linspace(400, 700, 31)
theta_cost = np.array([0])*np.pi/180 # 0: forward, 180: backward
phi_cost = np.array([0])*np.pi/180

# Wavelength & angle ranges over which to compute the final efficiencies and phase function
lam_plot = np.linspace(400, 700, 301)
theta_plot = np.array([0])*np.pi/180 # 0: forward, 180: backward
phi_plot = np.array([0])*np.pi/180

class cost_obj:
    def __init__(
            self,
            weight_abs,
            weight_sca,
            wvl_abs_tgt,
        ):
        
        self.weight_abs = weight_abs
        self.weight_sca = weight_sca
        self.wvl_abs_tgt = wvl_abs_tgt
    
    def cost(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml):
        cost = -self.weight_abs*np.sum(self.wvl_abs_tgt*np.log(Q_abs + 1e-8))\
              + self.weight_sca*np.sum((1 - self.wvl_abs_tgt)*np.log(Q_abs + 1e-8))
        
        return cost
    
    def gradient(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml, r):
        jac = -self.weight_abs*np.sum(self.wvl_abs_tgt[:,np.newaxis]*dQ_abs/(Q_abs[:,np.newaxis] + 1e-8), axis=0)\
             + self.weight_sca*np.sum((1 - self.wvl_abs_tgt)[:,np.newaxis]*dQ_abs/(Q_abs[:,np.newaxis] + 1e-8), axis=0)
    
        return jac

## Define Cost Function
weight_abs = 0
weight_sca = 1
wvl_abs_tgt = (lam_cost >= 650)*(lam_cost <= 650)

custom_cost = cost_obj(
    weight_abs=weight_abs,
    weight_sca=weight_sca,
    wvl_abs_tgt=wvl_abs_tgt,
    )

# Sweep Settings
r_min = 10
r_max = 1000
N_sweep = int((r_max - r_min)) + 1
d_low = 5
max_layers = None

if 'Ag_palik' in mat_needle:
    output_filename = directory + '/topopt_result_resonance_freq_lossy_650'
else:
    output_filename = directory + '/topopt_result_resonance_freq_lossless_650'

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
                    N_final=10,
                    verbose=0,
                    )
t2 = time.time()

max_mem_kb_proc = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
max_mem_kb = comm.reduce(max_mem_kb_proc, op=MPI.SUM, root=0)

if comm.rank == 0:
    print('### Time Elapsed: ' + str(np.round(t2 - t1, 3)) + ' s', flush=True)
    print('### Maximum Memory Usage: ' +str(np.round(max_mem_kb/1024**2, 3)) + ' GB', flush=True)