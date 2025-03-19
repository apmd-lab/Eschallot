import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, '/home/minseokhwan/')

import numpy as np
from scipy.ndimage import gaussian_filter1d
import Eschallot.optimization.topology_optimization as topopt
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD

import warnings
warnings.filterwarnings('ignore')

# Materials
mat_profile = np.array(['PMMA','TiO2_Sarkar']) # Outer(including embedding medium) to inner
mat_needle = np.array(['SiO2_bulk','TiO2_Sarkar']) # 'TiO2_Sarkar','Au_JC','Si_Schinke','Ag_palik', etc.

# Wavelength & angle ranges over which the cost function is defined
lam_cost = np.array([450])
theta_cost = np.linspace(0, 180, 361)*np.pi/180 # 0: forward, 180: backward
phi_cost = np.array([0,90])*np.pi/180

# Wavelength & angle ranges over which to compute the final efficiencies and phase function
lam_plot = np.array([450])
theta_plot = np.linspace(0, 180, 361)*np.pi/180 # 0: forward, 180: backward
phi_plot = np.array([0,90])*np.pi/180

class directional_reflector_cost:
    def __init__(self, ind_th_fwd, ind_th0, ind_th1, ind_phi_tgt, ind_wvl_tgt, theta):
        self.ind_th_fwd = ind_th_fwd
        self.ind_th0 = ind_th0
        self.ind_th1 = ind_th1
        self.ind_phi_tgt = ind_phi_tgt
        self.ind_wvl_tgt = ind_wvl_tgt
        self.theta = theta
        self.N_phi = ind_phi_tgt.size
    
    def cost(self, Q_sca, Q_abs, Q_ext, p, diff_CS):
        res = 0
        for i in range(self.N_phi):
            numer = np.sum(p[self.ind_wvl_tgt,self.ind_th0:self.ind_th1,self.ind_phi_tgt[i]]*np.sin(self.theta[self.ind_th0:self.ind_th1])[np.newaxis,:], axis=1)
            denom = np.sum(p[self.ind_wvl_tgt,self.ind_th_fwd:,:]*np.sin(self.theta[self.ind_th_fwd:])[np.newaxis,:,np.newaxis], axis=(1,2))
            res += -numer/denom
        
        return res
    
    def gradient(self, Q_sca, Q_abs, Q_ext, p, diff_CS, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, r):
        N_params = dQ_sca.shape[1]
    
        jac = np.zeros(N_params)
        for l in range(N_params):
            denom = np.sum(p[self.ind_wvl_tgt,self.ind_th_fwd:,:]*np.sin(self.theta[self.ind_th_fwd:])[np.newaxis,:,np.newaxis], axis=(1,2))
            d_denom = -np.sum(dp[self.ind_wvl_tgt,self.ind_th_fwd:,:,l]*np.sin(self.theta[self.ind_th_fwd:])[np.newaxis,:,np.newaxis], axis=(1,2))/denom**2
            for i in range(self.N_phi):
                numer = np.sum(p[self.ind_wvl_tgt,self.ind_th0:self.ind_th1,self.ind_phi_tgt[i]]*np.sin(self.theta[self.ind_th0:self.ind_th1])[np.newaxis,:], axis=1)
                d_numer = np.sum(dp[self.ind_wvl_tgt,self.ind_th0:self.ind_th1,self.ind_phi_tgt[i],l]*np.sin(self.theta[self.ind_th0:self.ind_th1])[np.newaxis,:], axis=1)
                
                jac[l] += -(d_numer/denom + numer*d_denom)
        
        return np.squeeze(jac)

# Define Cost Function
custom_cost = directional_reflector_cost(ind_th_fwd=11, # 5.5 deg (exclusive)
                                         ind_th0=291, # 145.5 deg (inclusive)
                                         ind_th1=302, # 151 deg (exclusive)
                                         ind_phi_tgt=np.array([0]), # 0 deg
                                         ind_wvl_tgt=0, # 450 nm
                                         theta=theta_cost)

# Sweep Settings
r_min = 10
r_max = 3000
N_sweep = int(r_max - r_min) + 1
d_low = 5
max_layers = None

output_filename = directory + '/topopt_result_data_theta148_phi_0_lambda450_rmax3000'

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
                    )