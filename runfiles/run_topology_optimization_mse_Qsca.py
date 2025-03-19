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
lam_cost = np.linspace(400, 550, 16)
theta_cost = np.linspace(0, 180, 2)*np.pi/180 # 0: forward, 180: backward
phi_cost = np.array([0,90])*np.pi/180

# Wavelength & angle ranges over which to compute the final efficiencies and phase function
lam_plot = np.linspace(400, 550, 301)
theta_plot = np.linspace(0, 180, 2)*np.pi/180 # 0: forward, 180: backward
phi_plot = np.array([0,90])*np.pi/180

class mse_cost:
    def __init__(self, Q_sca_con, Q_abs_con, Q_ext_con, p_con, diff_CS_con, N_lam, N_theta, N_phi):
        self.Q_sca_con = Q_sca_con
        self.Q_abs_con = Q_abs_con
        self.Q_ext_con = Q_ext_con
        self.p_con = p_con
        self.diff_CS_con = diff_CS_con
        self.N_lam = N_lam
        self.N_theta = N_theta
        self.N_phi = N_phi
    
    def cost(self, Q_sca, Q_abs, Q_ext, p, diff_CS):
        res = 0
    
        res += np.sum(self.Q_sca_con[0,1,:]*(Q_sca - self.Q_sca_con[0,0,:])**2)
        res += np.sum(self.Q_sca_con[1,1,:]*np.maximum(Q_sca - self.Q_sca_con[1,0,:], np.zeros(self.N_lam))**2)
        res += np.sum(self.Q_sca_con[2,1,:]*np.minimum(Q_sca - self.Q_sca_con[2,0,:], np.zeros(self.N_lam))**2)
        res += np.sum(self.Q_sca_con[3,1,:]*Q_sca)
        
        res += np.sum(self.Q_abs_con[0,1,:]*(Q_abs - self.Q_abs_con[0,0,:])**2)
        res += np.sum(self.Q_abs_con[1,1,:]*np.maximum(Q_abs - self.Q_abs_con[1,0,:], np.zeros(self.N_lam))**2)
        res += np.sum(self.Q_abs_con[2,1,:]*np.minimum(Q_abs - self.Q_abs_con[2,0,:], np.zeros(self.N_lam))**2)
        res += np.sum(self.Q_abs_con[3,1,:]*Q_abs)
        
        res += np.sum(self.Q_ext_con[0,1,:]*(Q_ext - self.Q_ext_con[0,0,:])**2)
        res += np.sum(self.Q_ext_con[1,1,:]*np.maximum(Q_ext - self.Q_ext_con[1,0,:], np.zeros(self.N_lam))**2)
        res += np.sum(self.Q_ext_con[2,1,:]*np.minimum(Q_ext - self.Q_ext_con[2,0,:], np.zeros(self.N_lam))**2)
        res += np.sum(self.Q_ext_con[3,1,:]*Q_ext)
        
        res += np.sum(self.p_con[0,1,:,:,:]*(p - self.p_con[0,0,:,:,:])**2)
        res += np.sum(self.p_con[1,1,:,:,:]*np.maximum(p - self.p_con[1,0,:,:,:], np.zeros((self.N_lam, self.N_theta, self.N_phi)))**2)
        res += np.sum(self.p_con[2,1,:,:,:]*np.minimum(p - self.p_con[2,0,:,:,:], np.zeros((self.N_lam, self.N_theta, self.N_phi)))**2)
        res += np.sum(self.p_con[3,1,:,:,:]*p)
        
        res += np.sum(self.diff_CS_con[3,1,:,:,:]*diff_CS)
    
        return res
    
    def gradient(self, Q_sca, Q_abs, Q_ext, p, diff_CS, dQ_sca, dQ_abs, dQ_ext, dp, d_diff_CS, r):
        N_params = dQ_sca.shape[1]
    
        jac = np.zeros(N_params)
        for l in range(N_params):
            jac[l] += np.sum(2*(Q_sca - self.Q_sca_con[0,0,:])*dQ_sca[:,l]*self.Q_sca_con[0,1,:])
            jac[l] += np.sum(2*np.maximum((Q_sca - self.Q_sca_con[1,0,:]), np.zeros(self.N_lam))*dQ_sca[:,l]*self.Q_sca_con[1,1,:])
            jac[l] += np.sum(2*np.minimum((Q_sca - self.Q_sca_con[2,0,:]), np.zeros(self.N_lam))*dQ_sca[:,l]*self.Q_sca_con[2,1,:])
            jac[l] += np.sum(dQ_sca[:,l]*self.Q_sca_con[3,1,:])
            
            jac[l] += np.sum(2*(Q_abs - self.Q_abs_con[0,0,:])*dQ_abs[:,l]*self.Q_abs_con[0,1,:])
            jac[l] += np.sum(2*np.maximum((Q_abs - self.Q_abs_con[1,0,:]), np.zeros(self.N_lam))*dQ_abs[:,l]*self.Q_abs_con[1,1,:])
            jac[l] += np.sum(2*np.minimum((Q_abs - self.Q_abs_con[2,0,:]), np.zeros(self.N_lam))*dQ_abs[:,l]*self.Q_abs_con[2,1,:])
            jac[l] += np.sum(dQ_abs[:,l]*self.Q_abs_con[3,1,:])
            
            jac[l] += np.sum(2*(Q_ext - self.Q_ext_con[0,0,:])*dQ_ext[:,l]*self.Q_ext_con[0,1,:])
            jac[l] += np.sum(2*np.maximum((Q_ext - self.Q_ext_con[1,0,:]), np.zeros(self.N_lam))*dQ_ext[:,l]*self.Q_ext_con[1,1,:])
            jac[l] += np.sum(2*np.minimum((Q_ext - self.Q_ext_con[2,0,:]), np.zeros(self.N_lam))*dQ_ext[:,l]*self.Q_ext_con[2,1,:])
            jac[l] += np.sum(dQ_ext[:,l]*self.Q_ext_con[3,1,:])
            
            jac[l] += np.sum(2*(p - self.p_con[0,0,:,:,:])*dp[:,:,:,l]*self.p_con[0,1,:,:,:])
            jac[l] += np.sum(2*np.maximum(p - self.p_con[1,0,:,:,:], np.zeros((self.N_lam, self.N_theta, self.N_phi)))*dp[:,:,:,l]*self.p_con[1,1,:,:,:])
            jac[l] += np.sum(2*np.minimum(p - self.p_con[2,0,:,:,:], np.zeros((self.N_lam, self.N_theta, self.N_phi)))*dp[:,:,:,l]*self.p_con[2,1,:,:,:])
            jac[l] += np.sum(dp[:,:,:,l]*self.p_con[3,1,:,:,:])
            
            jac[l] += np.sum(self.diff_CS_con[3,1,:,:,:]*d_diff_CS[:,:,:,l])
    
        return np.squeeze(jac)

## Define Cost Function
# 1st index: 0: equality, 1: lower than, 2: greater than, 3: weights only (non-least-squares/no target value)
# 2nd index: 0: target value, 1: weight
Q_sca_con = np.zeros((4, 2, lam_cost.size))
Q_abs_con = np.zeros((4, 2, lam_cost.size))
Q_ext_con = np.zeros((4, 2, lam_cost.size))
p_con = np.zeros((4, 2, lam_cost.size, theta_cost.size, phi_cost.size))
diff_CS_con = np.zeros((4, 2, lam_cost.size, theta_cost.size, phi_cost.size))

# Scattering Efficiency
np.random.seed(0)
Q_sca_tgt = gaussian_filter1d(np.random.rand(31), 3)
Q_sca_tgt = 2*(Q_sca_tgt - np.min(Q_sca_tgt))/np.ptp(Q_sca_tgt)
Q_sca_con[0,0,:] = Q_sca_tgt[:16]

np.savez(directory + '/mse_Q_sca_tgt', Q_sca_tgt=Q_sca_tgt)

# Sweep Settings
r_min = 10
r_max = 1000
N_sweep = int(r_max - r_min) + 1
d_low = 5
max_layers = None

# (1) Equal weights
Q_sca_con[0,1,:] = 1
Q_sca_con[0,1,5:11] = 1

custom_cost = mse_cost(Q_sca_con,
                       Q_abs_con,
                       Q_ext_con,
                       p_con,
                       diff_CS_con,
                       lam_cost.size,
                       theta_cost.size,
                       phi_cost.size)

output_filename = directory + '/topopt_result_mse_Q_sca_rmax1000_equal_weight'

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
                    )

# (2) Larger weights around the peak
Q_sca_con[0,1,:] = 1
Q_sca_con[0,1,5:11] = 5

custom_cost = mse_cost(Q_sca_con,
                       Q_abs_con,
                       Q_ext_con,
                       p_con,
                       diff_CS_con,
                       lam_cost.size,
                       theta_cost.size,
                       phi_cost.size)

output_filename = directory + '/topopt_result_mse_Q_sca_rmax1000_peak'

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
                    )