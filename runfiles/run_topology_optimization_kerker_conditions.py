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
            weight_tgt,
            weight_mag,
            weight_high_order,
            dpsi_tgt,
        ):
        
        self.weight_tgt = weight_tgt
        self.weight_mag = weight_mag
        self.weight_high_order = weight_high_order
        self.dpsi_tgt = dpsi_tgt
    
    def cost(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml):
        cost_tgt = np.abs(t_El[0,0] - t_Ml[0,0]*np.exp(1j*self.dpsi_tgt))**2
        cost_mag = -(np.abs(t_El[0,0])**2 + np.abs(t_Ml[0,0])**2)
        cost_high_order = np.sum(np.abs(t_El[:,1:])**2 + np.abs(t_Ml[:,1:])**2)
        
        cost = self.weight_tgt*cost_tgt + self.weight_mag*cost_mag + self.weight_high_order*cost_high_order
#        print('', flush=True)
#        print(cost, flush=True)
#        print(t_El[:,0], flush=True)
#        print(t_Ml[:,0], flush=True)
        
        return cost
    
    def gradient(self, Q_sca, Q_abs, Q_ext, p, diff_CS, t_El, t_Ml, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, dt_El, dt_Ml, r):
        jac_tgt = 2*(np.real(t_El[0,0] - t_Ml[0,0]*np.exp(1j*self.dpsi_tgt))*np.real(dt_El[0,0,:] - dt_Ml[0,0,:]*np.exp(1j*self.dpsi_tgt)) \
                   + np.imag(t_El[0,0] - t_Ml[0,0]*np.exp(1j*self.dpsi_tgt))*np.imag(dt_El[0,0,:] - dt_Ml[0,0,:]*np.exp(1j*self.dpsi_tgt)))
        
        jac_mag = 2*(np.real(t_El[0,0])*np.real(dt_El[0,0]) + np.imag(t_El[0,0])*np.imag(dt_El[0,0]) \
                   + np.real(t_Ml[0,0])*np.real(dt_Ml[0,0]) + np.imag(t_Ml[0,0])*np.imag(dt_Ml[0,0]))
        
        jac_high_order = 2*np.sum(np.real(np.expand_dims(t_El[:,1:], axis=-1))*np.real(dt_El[:,1:,:]) + np.imag(np.expand_dims(t_El[:,1:], axis=-1))*np.imag(dt_El[:,1:,:]) \
                                + np.real(np.expand_dims(t_Ml[:,1:], axis=-1))*np.real(dt_Ml[:,1:,:]) + np.imag(np.expand_dims(t_Ml[:,1:], axis=-1))*np.imag(dt_Ml[:,1:,:]))
        
        jac = self.weight_tgt*jac_tgt + self.weight_mag*jac_mag + self.weight_high_order*jac_high_order
        #print(jac, flush=True)
        #print(r, flush=True)
    
        return jac

## Define Cost Function
weight_tgt = 1e3
weight_mag = 0 #1e2
weight_high_order = 1e3
dpsi_tgt = np.pi/4

custom_cost = cost_obj(
    weight_tgt=weight_tgt,
    weight_mag=weight_mag,
    weight_high_order=weight_high_order,
    dpsi_tgt=dpsi_tgt,
)

# Sweep Settings
r_min = 10
r_max = 250
N_sweep = 5*int(r_max - r_min) + 1
d_low = 1.5
max_layers = None

if 'Ag_palik' in mat_needle:
    output_filename = directory + '/topopt_result_kerker_conditions_dpsi' + str(int(dpsi_tgt*180/np.pi))\
        + '_lossy_' + str(mat_profile[1]) + '_rmax' + str(r_max)\
        + '_weights' + str(weight_tgt) + '_' + str(weight_mag)  + '_' + str(weight_high_order)
else:
    output_filename = directory + '/topopt_result_kerker_conditions_dpsi' + str(int(dpsi_tgt*180/np.pi))\
        + '_lossless_' + str(mat_profile[1]) + '_rmax' + str(r_max)\
        + '_weights' + str(weight_tgt) + '_' + str(weight_mag)  + '_' + str(weight_high_order)

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