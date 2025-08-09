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
mat_needle = np.array(['TiO2_Sarkar','SiO2_bulk']) # 'TiO2_Sarkar','Au_JC','Si_Schinke','Ag_palik', etc.

# Wavelength & angle ranges over which the cost function is defined
lam_cost = np.array([450])
theta_cost = np.array([0])*np.pi/180 # 0: forward, 180: backward
phi_cost = np.array([0])*np.pi/180

# Wavelength & angle ranges over which to compute the final efficiencies and phase function
lam_plot = np.array([450])
theta_plot = np.linspace(0, 180, 361)*np.pi/180 # 0: forward, 180: backward
phi_plot = np.array([0,90])*np.pi/180

class cost_obj:
    def __init__(
            self,
            weight_fwd_diff,
            weight_fwd_mag,
            weight_bwd_diff,
            weight_bwd_mag,
            weight_high_order,
            ind_wvl_fwd,
            ind_wvl_bwd,
        ):
        
        self.weight_fwd_diff = weight_fwd_diff
        self.weight_fwd_mag = weight_fwd_mag
        self.weight_bwd_diff = weight_bwd_diff
        self.weight_bwd_mag = weight_bwd_mag
        self.weight_high_order = weight_high_order
        self.ind_wvl_fwd = ind_wvl_fwd
        self.ind_wvl_bwd = ind_wvl_bwd
    
    def cost(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml):
        cost_fwd_diff = 0
        cost_fwd_mag = 0
        if self.ind_wvl_fwd is not None:
            for i in self.ind_wvl_fwd:
                cost_fwd_diff += np.abs(t_El[i,0] - t_Ml[i,0])**2
                cost_fwd_mag -= np.abs(t_El[i,0])**2 + np.abs(t_Ml[i,0])**2
        cost_fwd = self.weight_fwd_diff*cost_fwd_diff + self.weight_fwd_mag*cost_fwd_mag
        
        cost_bwd_diff = 0
        cost_bwd_mag = 0
        if self.ind_wvl_bwd is not None:
            for i in self.ind_wvl_bwd:
                cost_bwd_diff += np.abs(t_El[i,0] + t_Ml[i,0])**2
                cost_bwd_mag -= np.abs(t_El[i,0])**2 + np.abs(t_Ml[i,0])**2
        cost_bwd = self.weight_bwd_diff*cost_bwd_diff + self.weight_bwd_mag*cost_bwd_mag
        
        cost_high_order = np.sum(np.abs(t_El[:,1:])**2 + np.abs(t_Ml[:,1:])**2)
        
        cost = cost_fwd + cost_bwd + self.weight_high_order*cost_high_order
#        print('', flush=True)
#        print(cost, flush=True)
#        print(t_El[:,:2], flush=True)
#        print(t_Ml[:,:2], flush=True)
        
        return cost
    
    def gradient(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml, r):
        jac_fwd_diff = np.zeros(r.size)
        jac_fwd_mag = np.zeros(r.size)
        if self.ind_wvl_fwd is not None:
            for i in self.ind_wvl_fwd:
                jac_fwd_diff += 2*(np.real(t_El[i,0] - t_Ml[i,0])*np.real(dt_El[i,0,:] - dt_Ml[i,0,:]) \
                                 + np.imag(t_El[i,0] - t_Ml[i,0])*np.imag(dt_El[i,0,:] - dt_Ml[i,0,:]))
                jac_fwd_mag -= 2*(np.real(t_El[i,0])*np.real(dt_El[i,0,:]) + np.imag(t_El[i,0])*np.imag(dt_El[i,0,:]) \
                                + np.real(t_Ml[i,0])*np.real(dt_Ml[i,0,:]) + np.imag(t_Ml[i,0])*np.imag(dt_Ml[i,0,:]))
        jac_fwd = self.weight_fwd_diff*jac_fwd_diff + self.weight_fwd_mag*jac_fwd_mag
        
        jac_bwd_diff = np.zeros(r.size)
        jac_bwd_mag = np.zeros(r.size)
        if self.ind_wvl_bwd is not None:
            for i in self.ind_wvl_bwd:
                jac_bwd_diff += 2*(np.real(t_El[i,0] + t_Ml[i,0])*np.real(dt_El[i,0,:] + dt_Ml[i,0,:]) \
                                 + np.imag(t_El[i,0] + t_Ml[i,0])*np.imag(dt_El[i,0,:] + dt_Ml[i,0,:]))
                jac_bwd_mag -= 2*(np.real(t_El[i,0])*np.real(dt_El[i,0,:]) + np.imag(t_El[i,0])*np.imag(dt_El[i,0,:]) \
                                + np.real(t_Ml[i,0])*np.real(dt_Ml[i,0,:]) + np.imag(t_Ml[i,0])*np.imag(dt_Ml[i,0,:]))
        jac_bwd = self.weight_bwd_diff*jac_bwd_diff + self.weight_bwd_mag*jac_bwd_mag
        
        jac_high_order = 2*np.sum(np.real(t_El[:,1:])*np.real(dt_El[:,1:,:]) + np.imag(t_El[:,1:])*np.imag(dt_El[:,1:,:]) \
                                + np.real(t_Ml[:,1:])*np.real(dt_Ml[:,1:,:]) + np.imag(t_Ml[:,1:])*np.imag(dt_Ml[:,1:,:]))
        
        jac = jac_fwd + jac_bwd + self.weight_high_order*jac_high_order
        #print(r, flush=True)
    
        return jac

## Define Cost Function
weight_fwd_diff = 1e3
weight_fwd_mag = 1e2
weight_bwd_diff = 1e3
weight_bwd_mag = 1e2
weight_high_order = 1e2
ind_wvl_fwd = [0]
ind_wvl_bwd = None

custom_cost = cost_obj(
    weight_fwd_diff=weight_fwd_diff,
    weight_fwd_mag=weight_fwd_mag,
    weight_bwd_diff=weight_bwd_diff,
    weight_bwd_mag=weight_bwd_mag,
    weight_high_order=weight_high_order,
    ind_wvl_fwd=ind_wvl_fwd,
    ind_wvl_bwd=ind_wvl_bwd,
    )

# Sweep Settings
r_min = 10
r_max = 250
N_sweep = int(r_max - r_min) + 1
d_low = 5
max_layers = None

if 'Ag_palik' in mat_needle:
    output_filename = directory + '/topopt_result_kerker_conditions_fwd' + str(ind_wvl_fwd) + '_bwd' + str(ind_wvl_bwd) + '_lossy_' + str(mat_profile[1]) + '_rmax' + str(r_max)
else:
    output_filename = directory + '/topopt_result_kerker_conditions_fwd' + str(ind_wvl_fwd) + '_bwd' + str(ind_wvl_bwd) + '_lossless_' + str(mat_profile[1]) + '_rmax' + str(r_max)

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
                    verbose=True,
                    )
t2 = time.time()

max_mem_kb_proc = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
max_mem_kb = comm.reduce(max_mem_kb_proc, op=MPI.SUM, root=0)

if comm.rank == 0:
    print('### Time Elapsed: ' + str(np.round(t2 - t1, 3)) + ' s', flush=True)
    print('### Maximum Memory Usage: ' +str(np.round(max_mem_kb/1024**2, 3)) + ' GB', flush=True)