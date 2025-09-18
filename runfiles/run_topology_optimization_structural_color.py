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
mat_profile = np.array(['Air','Ag_palik']) # Outer(including embedding medium) to inner
mat_needle = np.array(['TiO2_Sarkar','SiO2_bulk','Au_JC','Si_Schinke','Ag_palik']) # 'TiO2_Sarkar','Au_JC','Si_Schinke','Ag_palik', etc.

# Wavelength & angle ranges over which the cost function is defined
lam_cost = np.linspace(400, 700, 301)
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
            weight_trans,
            wvl_trans_tgt,
            abs_threshold,
        ):
        
        self.weight_abs = weight_abs
        self.weight_trans = weight_trans
        self.wvl_trans_tgt = wvl_trans_tgt
        self.abs_threshold = abs_threshold
    
    def cost(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml):
        if np.sum(Q_abs < 0) > 0:
            x = np.arange(Q_abs.size)
            x_interp = np.zeros(0)
            for i in range(Q_abs.size):
                if Q_abs[i] < 0:
                    x_interp = np.append(x_interp, i)
            Q_abs[list(x_interp)] = np.interp(x_interp, x, Q_abs)
            np.savez(directory + '/debug_cost', Q_abs=Q_abs)
    
        cost_trans = self.weight_trans*np.sum(self.wvl_trans_tgt*np.log(Q_abs + 1e-8))
        cost_abs_log = (1 - self.wvl_trans_tgt)*np.log(Q_abs + 1e-8)
        cost_abs = -self.weight_abs*np.sum(cost_abs_log/(1 + np.exp(10*(cost_abs_log - self.abs_threshold))))/np.sum(1 - self.wvl_trans_tgt)
        
        cost = cost_trans + cost_abs
        
        if cost == np.inf or np.isnan(cost):
            np.savez(directory + '/debug_cost', Q_abs=Q_abs)
#            print(cost_trans, flush=True)
#            print(cost_abs, flush=True)
        
        return cost
    
    def gradient(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml, r):
        if np.sum(Q_abs < 0) > 0:
            x = np.arange(Q_abs.size)
            x_interp = np.zeros(0)
            for i in range(Q_abs.size):
                if Q_abs[i] < 0:
                    x_interp = np.append(x_interp, i)
            Q_abs[list(x_interp)] = np.interp(x_interp, x, Q_abs)
    
        cost_abs_log = (1 - self.wvl_trans_tgt)*np.log(Q_abs + 1e-8)
        logistic_fct = 1/(1 + np.exp(10*(cost_abs_log - self.abs_threshold)))
        
        jac_trans = self.weight_trans*np.sum(self.wvl_trans_tgt[:,np.newaxis]*dQ_abs/(Q_abs[:,np.newaxis] + 1e-8), axis=0)
        jac_abs = -(self.weight_abs/np.sum(1 - self.wvl_trans_tgt[:,np.newaxis]))\
                  *np.sum((logistic_fct[:,np.newaxis]/(Q_abs[:,np.newaxis] + 1e-8))\
                          *(1 - 10*np.exp(10*(cost_abs_log[:,np.newaxis] - self.abs_threshold))*logistic_fct[:,np.newaxis])*dQ_abs, axis=0)
    
        jac = jac_trans + jac_abs
        
        if np.sum(jac == np.inf) > 0 or np.sum(np.isnan(jac)) > 0:
            np.savez(directory + '/debug_cost', Q_abs=Q_abs, dQ_abs=dQ_abs, r=r)
#            print(jac_trans, flush=True)
#            print(jac_abs, flush=True)
    
        return jac

## Define Cost Function
weight_abs = 1
weight_trans = 1
wvl_trans_tgt = (lam_cost >= 650)*(lam_cost <= 650)
abs_threshold = -1

custom_cost = cost_obj(
    weight_abs=weight_abs,
    weight_trans=weight_trans,
    wvl_trans_tgt=wvl_trans_tgt,
    abs_threshold=abs_threshold,
    )

# Sweep Settings
r_min = 10
r_max = 250
N_sweep = int((r_max - r_min)) + 1
d_low = 5
max_layers = None

if 'Ag_palik' in mat_needle:
    output_filename = directory + '/topopt_result_structural_color_lossy_650_rmax' + str(r_max)
else:
    output_filename = directory + '/topopt_result_structural_color_lossless_650_rmax' + str(r_max)

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
                    N_final=100,
                    verbose=0,
                    )
t2 = time.time()

max_mem_kb_proc = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
max_mem_kb = comm.reduce(max_mem_kb_proc, op=MPI.SUM, root=0)

if comm.rank == 0:
    print('### Time Elapsed: ' + str(np.round(t2 - t1, 3)) + ' s', flush=True)
    print('### Maximum Memory Usage: ' +str(np.round(max_mem_kb/1024**2, 3)) + ' GB', flush=True)