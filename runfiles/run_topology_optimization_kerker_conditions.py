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
lam_cost = np.array([650])
theta_cost = np.array([0])*np.pi/180 # 0: forward, 180: backward
phi_cost = np.array([0])*np.pi/180

# Wavelength & angle ranges over which to compute the final efficiencies and phase function
lam_plot = np.array([650])
theta_plot = np.linspace(0, 180, 361)*np.pi/180 # 0: forward, 180: backward
phi_plot = np.array([0,90])*np.pi/180

class cost_obj:
    def __init__(
            self,
            weight_fwd,
            weight_bwd,
            weight_bwd_mag,
            weight_high_order,
            ind_wvl_fwd,
            ind_wvl_bwd,
        ):
        
        self.weight_fwd = weight_fwd
        self.weight_bwd = weight_bwd
        self.weight_bwd_mag = weight_bwd_mag
        self.weight_high_order = weight_high_order
        self.ind_wvl_fwd = ind_wvl_fwd
        self.ind_wvl_bwd = ind_wvl_bwd
    
    def cost(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml):
        cost_fwd = 0
        if self.ind_wvl_fwd is not None:
            for i in self.ind_wvl_fwd:
                cost_fwd += np.abs(t_El[i,0] - t_Ml[i,0])**2
        
        cost_bwd = 0
        cost_bwd_mag = 0
        if self.ind_wvl_bwd is not None:
            for i in self.ind_wvl_bwd:
                cost_bwd += np.abs(t_El[i,0] + t_Ml[i,0])**2
                cost_bwd_mag -= np.abs(t_El[i,0])**2 + np.abs(t_Ml[i,0])**2
        
        cost_high_order = np.sum(np.abs(t_El[:,1:])**2 + np.abs(t_Ml[:,1:])**2)
        
        cost = self.weight_fwd*cost_fwd + self.weight_bwd*cost_bwd + self.weight_bwd_mag*cost_bwd_mag + self.weight_high_order*cost_high_order
#        print('', flush=True)
#        print(cost, flush=True)
#        print(t_El[:,0], flush=True)
#        print(t_Ml[:,0], flush=True)
        
        return cost
    
    def gradient(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml, r):
        dim_gradient = dQ_sca.shape[1]
        jac_fwd = np.zeros(dim_gradient)
        if self.ind_wvl_fwd is not None:
            for i in self.ind_wvl_fwd:
                jac_fwd += 2*(np.real(t_El[i,0] - t_Ml[i,0])*np.real(dt_El[i,0,:] - dt_Ml[i,0,:]) \
                            + np.imag(t_El[i,0] - t_Ml[i,0])*np.imag(dt_El[i,0,:] - dt_Ml[i,0,:]))
        
        jac_bwd = np.zeros(dim_gradient)
        jac_bwd_mag = np.zeros(dim_gradient)
        if self.ind_wvl_bwd is not None:
            for i in self.ind_wvl_bwd:
                jac_bwd += 2*(np.real(t_El[i,0] + t_Ml[i,0])*np.real(dt_El[i,0,:] + dt_Ml[i,0,:]) \
                            + np.imag(t_El[i,0] + t_Ml[i,0])*np.imag(dt_El[i,0,:] + dt_Ml[i,0,:]))
                jac_bwd_mag -= 2*(np.real(t_El[i,0])*np.real(dt_El[i,0]) + np.imag(t_El[i,0])*np.imag(dt_El[i,0]) \
                                + np.real(t_Ml[i,0])*np.real(dt_Ml[i,0]) + np.imag(t_Ml[i,0])*np.imag(dt_Ml[i,0]))
        
        jac_high_order = 2*np.sum(np.real(np.expand_dims(t_El[:,1:], axis=-1))*np.real(dt_El[:,1:,:]) + np.imag(np.expand_dims(t_El[:,1:], axis=-1))*np.imag(dt_El[:,1:,:]) \
                                + np.real(np.expand_dims(t_Ml[:,1:], axis=-1))*np.real(dt_Ml[:,1:,:]) + np.imag(np.expand_dims(t_Ml[:,1:], axis=-1))*np.imag(dt_Ml[:,1:,:]))
        
        jac = self.weight_fwd*jac_fwd + self.weight_bwd*jac_bwd + self.weight_bwd_mag*jac_bwd_mag + self.weight_high_order*jac_high_order
        #print(jac, flush=True)
        #print(r, flush=True)
    
        return jac

## Define Cost Function
weight_fwd = 0
weight_bwd = 1e3
weight_bwd_mag = 1e2
weight_high_order = 1e2
ind_wvl_fwd = None
ind_wvl_bwd = [0]

custom_cost = cost_obj(
    weight_fwd=weight_fwd,
    weight_bwd=weight_bwd,
    weight_bwd_mag=weight_bwd_mag,
    weight_high_order=weight_high_order,
    ind_wvl_fwd=ind_wvl_fwd,
    ind_wvl_bwd=ind_wvl_bwd,
    )

# Sweep Settings
r_min = 10
r_max = 250
N_sweep = 5*int(r_max - r_min) + 1
d_low = 1.5
max_layers = None

if 'Ag_palik' in mat_needle:
    output_filename = directory + '/topopt_result_kerker_conditions_fwd' + str(lam_cost[ind_wvl_fwd])\
        + '_bwd' + str(lam_cost[ind_wvl_bwd]) + '_lossy_' + str(mat_profile[1]) + '_rmax' + str(r_max)\
        + '_weights' + str(weight_fwd) + '_' + str(weight_bwd)  + '_' + str(weight_high_order)
else:
    output_filename = directory + '/topopt_result_kerker_conditions_fwd' + str(lam_cost[ind_wvl_fwd])\
        + '_bwd' + str(lam_cost[ind_wvl_bwd]) + '_lossless_' + str(mat_profile[1]) + '_rmax' + str(r_max)\
        + '_weights' + str(weight_fwd) + '_' + str(weight_bwd) + '_' + str(weight_high_order)

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
                    N_final=200,
                    verbose=1,
                    )
t2 = time.time()

max_mem_kb_proc = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
max_mem_kb = comm.reduce(max_mem_kb_proc, op=MPI.SUM, root=0)

if comm.rank == 0:
    print('### Time Elapsed: ' + str(np.round(t2 - t1, 3)) + ' s', flush=True)
    print('### Maximum Memory Usage: ' +str(np.round(max_mem_kb/1024**2, 3)) + ' GB', flush=True)